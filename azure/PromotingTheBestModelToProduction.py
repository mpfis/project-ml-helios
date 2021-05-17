# Databricks notebook source
# MAGIC %md
# MAGIC # Promoting the Best Model to Production
# MAGIC 
# MAGIC The idea of this notebook is to compare an existing model in Production to a newly retrained model in Staging and promote the Staging model to Production if it performs better. The general workflow is illustrated below.
# MAGIC 
# MAGIC <img width="800" src='/files/promoting_model_workflow.png'/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare the models in Production and Staging

# COMMAND ----------

import mlflow
from sklearn.metrics import mean_squared_error,  r2_score

import urllib 
import json 

# COMMAND ----------

class ModelEvaluation:
    def __init__(self, model_name, env):
        self.client = mlflow.tracking.MlflowClient()
        self.latest_model_details = self.client.get_latest_versions(
            model_name, stages=[env])[0]
        
    def get_training_metrics(self, *, training_metrics: list):
        """
        Get model metrics calculated during training of the model
        ...

        Parameters
        ----------
        training_metrics: list(str)
            List of training metrics as called in MLFlow, eg ['training_score', 'training_rmse'] 
        
        Returns
        -------
        A dictionary of training metric names and their values for the given model in the given environment
        """
        training_metrics_results = {}
        for training_metric in training_metrics:
            training_metrics_result = self.client.get_run(
                self.latest_model_details.run_id
            ).data.metrics[training_metric]
            training_metrics_results[training_metric] = training_metrics_result
        return training_metrics_results

    def get_performance_on_test_dataset(self, *, test_feats, test_label, test_evaluation_methods: list):
        '''
        Get performance metrics of the model on the test dataset
        
        Parameters
        ----------
        test_feats: pd.DataFrame()
            A pandas Dataframe with test dataset features
            
        test_label: pd.DataFrame()
            A pandas Dataframe with test dataset labels
        
        test_evaluation_methods: a list of dictionaries 
            Each element in the list is a dictionary containing a 'func' key which is a method to evaluate how good the model's predictions are compared to the test labels and a 'params' key which are the necessary parameters for 'func' method, Eg. [ {"func": mean_squared_error, "params": {"squared": False}}]
        
        Returns
        -------
        A dictionary of test metric names and their values for the given model in the given environment
        '''
        # Load model as a PyFuncModel to score the test data
        loaded_model = mlflow.pyfunc.load_model(self.latest_model_details.source)
        # Predict on test data
        test_df_predictions = loaded_model.predict(test_df_feats)
        test_metrics_results = {}
        for method in test_evaluation_methods:
            method_result =  method['func'](test_df_label.values, test_df_predictions, **method['params'])
            test_metrics_results[str(method['func'].__name__)] = method_result
        return test_metrics_results


# COMMAND ----------

test_raw_df = spark.read.format('delta').load("dbfs:/sepideh_ebrahimi/diamonds/test_data")
test_df = test_raw_df.toPandas()
test_df_feats = test_df[['x', 'y', 'z', 'carat_squared']]
test_df_label = test_df[['price']]

# COMMAND ----------

model_name = "diamonds_price_model"
training_metrics=["training_score", "training_rmse", "training_r2_score"]
test_evaluation_methods=[
        {"func": mean_squared_error, "params": {"squared": False}},
        {"func": r2_score, "params": {}}
    ]

production_model_evaluation = ModelEvaluation(model_name=model_name, env="Production")

# Get training metrics of the Production model
training_metrics_production = production_model_evaluation.get_training_metrics(
    training_metrics=training_metrics
)

# Get test metrics of the Production model
test_metrics_production = production_model_evaluation.get_performance_on_test_dataset(
    test_feats=test_df_feats,
    test_label=test_df_label,
    test_evaluation_methods=test_evaluation_methods
)


# COMMAND ----------

staging_model_evaluation = ModelEvaluation(model_name=model_name, env="Staging")

# Get training metrics of the Staging model
training_metrics_staging = staging_model_evaluation.get_training_metrics(
    training_metrics=training_metrics
)

# Get test metrics of the Staging model
test_metrics_staging = staging_model_evaluation.get_performance_on_test_dataset(
    test_feats=test_df_feats,
    test_label=test_df_label,
    test_evaluation_methods=test_evaluation_methods
)

# COMMAND ----------

print(training_metrics_production)
print(test_metrics_production)

# COMMAND ----------

print(training_metrics_staging)
print(test_metrics_staging)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Raise a Slack alert if the Staging model is better

# COMMAND ----------

def raise_a_Slack_alert(model_name):
    slack_token = dbutils.secrets.get("sepideh-model-monitoring", "slack-verification-token")
    slack_webhook = dbutils.secrets.get("sepideh-model-monitoring", "slack-app-url")
    
    body = {
    "text": f"There is a new version of {model_name} model ready to move to production!"
    }
    
    json_body = json.dumps(body) 
    json_bytes = json_body.encode('utf-8') 
    headers = { 'Authorization': f'Bearer {slack_token}'} 

    payload = urllib.request.Request(slack_webhook, data=json_bytes, headers=headers)  
    response = urllib.request.urlopen(payload) 
    return response

# COMMAND ----------

training_metrics_lower_is_better = ["training_rmse"]
training_metrics_higher_is_better = ["training_score", "training_r2_score"]

test_metrics_lower_is_better = ["mean_squared_error"]
test_metrics_higher_is_better = ["r2_score"]

def compare_Staging_and_Production_models(*, training_metrics_production, test_metrics_production, training_metrics_staging, test_metrics_staging):
    is_staging_training_metrics_better = True
    is_staging_test_metrics_better = True
    # Compare trainig metrics
    for k in training_metrics_production.keys():
        if k in training_metrics_lower_is_better:
            is_staging_training_metrics_better = is_staging_training_metrics_better and (training_metrics_staging[k] < training_metrics_production[k])
        elif k in training_metrics_higher_is_better:
            is_staging_training_metrics_better = is_staging_training_metrics_better and (training_metrics_staging[k] > training_metrics_production[k])
    
    # Compare test metrics
    for k in test_metrics_production.keys():
        if k in test_metrics_lower_is_better:
            is_staging_test_metrics_better = is_staging_test_metrics_better and (test_metrics_staging[k] < test_metrics_production[k])
        elif k in test_metrics_higher_is_better:
            is_staging_test_metrics_better = is_staging_test_metrics_better and (test_metrics_staging[k] > test_metrics_production[k])
            
    if is_staging_training_metrics_better and is_staging_test_metrics_better:
        response = raise_a_Slack_alert(model_name)
        

# COMMAND ----------

compare_Staging_and_Production_models(
    training_metrics_production=training_metrics_production,
    test_metrics_production=test_metrics_production,
    training_metrics_staging=training_metrics_staging,
    test_metrics_staging=test_metrics_staging,
)

# COMMAND ----------

# client.transition_model_version_stage(
#     model_name,
#     version=staging_model_evaluation.latest_model_details.version,
#     stage="Production",
#     archive_existing_versions=False,
# )

# client.transition_model_version_stage(
#     model_name,
#     version=production_model_evaluation.latest_model_details.version,
#     stage="Archived",
#     archive_existing_versions=False,
# )

# COMMAND ----------


