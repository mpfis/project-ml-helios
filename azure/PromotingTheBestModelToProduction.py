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

# COMMAND ----------

test_raw_df = spark.read.format('delta').load("dbfs:/sepideh_ebrahimi/diamonds/test_data")
test_df = test_raw_df.toPandas()
test_df_feats = test_df[['x', 'y', 'z', 'carat_squared']]
test_df_label = test_df[['price']]

# COMMAND ----------

from sklearn.metrics import mean_squared_error

client = mlflow.tracking.MlflowClient()

def evaluate_model(test_df_feats, test_df_label, env="Production", model_name="diamonds_price_model"):
    # Get the training score of the production model from model registry
    latest_model_details = client.get_latest_versions(model_name, stages=[env])[0]
    training_score = client.get_run(latest_model_details.run_id).data.metrics[
        "training_score"
    ]

    # Load model as a PyFuncModel to score the test data
    loaded_model = mlflow.pyfunc.load_model(latest_model_details.source)

    # Predict on test data
    test_df_predictions = loaded_model.predict(test_df_feats)

    rmse = mean_squared_error(test_df_label.values, test_df_predictions, squared=False)

    return (latest_model_details, training_score, rmse)


# COMMAND ----------

latest_model_details_prod, training_score_prod, rmse_prod = evaluate_model(test_df_feats, test_df_label, env="Production", model_name="diamonds_price_model")

latest_model_details_staging, training_score_staging, rmse_staging = evaluate_model(test_df_feats, test_df_label, env="Staging", model_name="diamonds_price_model")

# COMMAND ----------

if training_score_staging > training_score_prod and rmse_staging < rmse_prod:
    client.transition_model_version_stage(
        model_name,
        version=latest_model_details_staging.version,
        stage="Production",
        archive_existing_versions=False,
    )
  
    client.transition_model_version_stage(
        model_name,
        version=latest_model_details_prod.version,
        stage="Archived",
        archive_existing_versions=False,
    )

# COMMAND ----------


