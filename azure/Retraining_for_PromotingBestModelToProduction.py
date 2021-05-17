# Databricks notebook source
# MAGIC %md
# MAGIC # Workflow
# MAGIC 
# MAGIC     
# MAGIC * Read in the data and transform and create features
# MAGIC * Train a model and move it to production
# MAGIC * Train a second/third model and move it to staging

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading in data and transformations

# COMMAND ----------

diamonds_df_raw = spark.read.csv(
    "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv",
    header="true",
    inferSchema="true",
)

# COMMAND ----------

from pyspark.ml.feature import Imputer

imputer = Imputer(
    inputCols=["x", "y", "z", "carat", "price"],
    outputCols=["{}".format(c) for c in ["x", "y", "z", "carat", "price"]],
)
imputer.fit(diamonds_df_raw).transform(diamonds_df_raw)

# COMMAND ----------

display(diamonds_df_raw.summary())

# COMMAND ----------

from pyspark.sql.functions import udf, col
from pyspark.sql.types import DoubleType

# Define a power of 2 udf
pow_2_udf = udf(lambda x: x * x, DoubleType())
diamonds_df = diamonds_df_raw.select(
    "price", "x", "y", "z", pow_2_udf("carat").alias("carat_squared")
)

# COMMAND ----------

train_and_validation_raw_df, test_raw_df = diamonds_df.randomSplit([0.8, 0.2], seed=42)

# Save the test data for later evaluation of models
test_raw_df.write.format("delta").mode("overwrite").save(
    "dbfs:/sepideh_ebrahimi/diamonds/test_data"
)
train_and_validation_raw_df.write.format("delta").mode("overwrite").save(
    "dbfs:/sepideh_ebrahimi/diamonds/train_and_validation_data"
)

# COMMAND ----------

from sklearn.model_selection import train_test_split

train_and_validation_df = train_and_validation_raw_df.toPandas()[
    ["x", "y", "z", "carat_squared", "price"]
]

x_train, x_test, y_train, y_test = train_test_split(
    train_and_validation_df.drop("price", axis=1),
    train_and_validation_df["price"],
    test_size=0.3,
    random_state=42,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train a first (Linear Regression) model and log the experiment

# COMMAND ----------

import mlflow
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint

# enable autologging
mlflow.sklearn.autolog()


def fetch_logged_data(run_id):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts


with mlflow.start_run(run_name="Linear Regression") as lr_run:
   
    # Train a model
    lr_model = LinearRegression()
    lr_model.fit(x_train, y_train)

    # Predict 
    predictions = lr_model.predict(x_test)
    predictions = pd.concat(
        [
            x_test.reset_index(drop=True),
            y_test.reset_index(drop=True),
            pd.Series(predictions),
        ],
        axis=1,
    )
    predictions = predictions.rename(columns={0: "prediction"})
    predictions["carat"] = predictions["carat_squared"] ** 2
    predictions = predictions.sort_values("carat")
    
    # Plot the predictions against the carat feature
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.scatter(predictions.carat, predictions.price, color="royalblue")
    ax.plot(
        predictions.carat,
        predictions.prediction,
        color="r",
        label="linear regression",
        linewidth=3,
    )

    ax.set_xlabel("carat", fontsize=12)
    ax.set_ylabel("price", fontsize=12)
    ax.set_title("Linear Regression", fontsize=16)
    ax.legend()
    plt.savefig("regression_plot.png", bbox_inches="tight")
    plt.close()
    
    # Log the plot
    mlflow.log_artifact("regression_plot.png")

    params, metrics, tags, artifacts = fetch_logged_data(lr_run.info.run_id)
    pprint(params)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the first trained model in Production

# COMMAND ----------

import time

model_name = "diamonds_price_model"
client = mlflow.tracking.MlflowClient()
try:
    client.create_registered_model(model_name)
except Exception as e:
    pass

model_version = client.create_model_version(
    model_name, f"{lr_run.info.artifact_uri}/model", lr_run.info.run_id
)

time.sleep(3)  # Just to make sure it's had a second to register
client.update_model_version(
    model_name, model_version.version, description="Current candidate"
)

# COMMAND ----------

# client.transition_model_version_stage(model_name, version=model_version.version, stage="Staging", archive_existing_versions=False)

client.transition_model_version_stage(model_name, version=model_version.version, stage="Production", archive_existing_versions=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train a second (Random Forest) model

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor

with mlflow.start_run(run_name="Random Forest") as rf_run:
    # Train a model
    rf_model = RandomForestRegressor(min_samples_leaf=10, min_samples_split=10)
    rf_model.fit(x_train, y_train)
  
    # Predict 
    predictions = rf_model.predict(x_test)
    predictions = pd.concat(
        [
            x_test.reset_index(drop=True),
            y_test.reset_index(drop=True),
            pd.Series(predictions),
        ],
        axis=1,
    )
    predictions = predictions.rename(columns={0: "prediction"})
    predictions["carat"] = predictions["carat_squared"] ** 2
    predictions = predictions.sort_values("carat")
    
     # Plot the predictions against the carat feature
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.scatter(predictions.carat, predictions.price, color="royalblue")
    ax.plot(
        predictions.carat,
        predictions.prediction,
        color="r",
        label="Random Forest",
        linewidth=3,
    )

    ax.set_xlabel("carat", fontsize=12)
    ax.set_ylabel("price", fontsize=12)
    ax.set_title("Random Forest", fontsize=16)
    ax.legend()
    plt.savefig("regression_plot.png", bbox_inches="tight")
    plt.close()
    
    # Log the plot
    mlflow.log_artifact("regression_plot.png")

    params, metrics, tags, artifacts = fetch_logged_data(rf_run.info.run_id)
    pprint(params)
  

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the second trained model in Staging

# COMMAND ----------

try:
    client.create_registered_model(model_name)
except Exception as e:
    pass

model_version = client.create_model_version(
    model_name, f"{rf_run.info.artifact_uri}/model", rf_run.info.run_id
)

time.sleep(3)  # Just to make sure it's had a second to register
client.update_model_version(
    model_name, model_version.version, description="Current candidate"
)

# COMMAND ----------

client.transition_model_version_stage(model_name, version=model_version.version, stage="Staging", archive_existing_versions=False)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Train a third (SVR) model

# COMMAND ----------

# from sklearn.svm import SVR

# with mlflow.start_run(run_name="SVR") as svr_run:
#     # Train a model
#     svr_model = SVR()
#     svr_model.fit(x_train, y_train)
  
#     # Predict 
#     predictions = svr_model.predict(x_test)
#     predictions = pd.concat(
#         [
#             x_test.reset_index(drop=True),
#             y_test.reset_index(drop=True),
#             pd.Series(predictions),
#         ],
#         axis=1,
#     )
#     predictions = predictions.rename(columns={0: "prediction"})
#     predictions["carat"] = predictions["carat_squared"] ** 2
#     predictions = predictions.sort_values("carat")
    
#      # Plot the predictions against the carat feature
#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))

#     ax.scatter(predictions.carat, predictions.price, color="royalblue")
#     ax.plot(
#         predictions.carat,
#         predictions.prediction,
#         color="r",
#         label="SVR",
#         linewidth=3,
#     )

#     ax.set_xlabel("carat", fontsize=12)
#     ax.set_ylabel("price", fontsize=12)
#     ax.set_title("SVR", fontsize=16)
#     ax.legend()
#     plt.savefig("regression_plot.png", bbox_inches="tight")
#     plt.close()
    
#     # Log the plot
#     mlflow.log_artifact("regression_plot.png")

#     params, metrics, tags, artifacts = fetch_logged_data(svr_run.info.run_id)
#     pprint(params)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the third trained model in Staging

# COMMAND ----------

# try:
#     client.create_registered_model(model_name)
# except Exception as e:
#     pass

# model_version = client.create_model_version(
#     model_name, f"{svr_run.info.artifact_uri}/model", svr_run.info.run_id
# )

# time.sleep(3)  # Just to make sure it's had a second to register
# client.update_model_version(
#     model_name, model_version.version, description="Current candidate"
# )

# COMMAND ----------

# client.transition_model_version_stage(model_name, version=model_version.version, stage="Staging", archive_existing_versions=False)

# COMMAND ----------


