# Databricks notebook source
dbutils.widgets.text("resource_group", "")
dbutils.widgets.text("endpointName", "")
dbutils.widgets.text("app_insights_name", "")
dbutils.widgets.text("PATH_TO_MLFLOW_EXPERIMENT", "")
dbutils.widgets.text("hours", "")
dbutils.widgets.text("DB_NAME", "")
dbutils.widgets.text("secretScope", "")

# COMMAND ----------

scopeName = dbutils.widgets.get("secretScope")
clientId = dbutils.secrets.get(f"{scopeName}", "spId")
clientSecret = dbutils.secrets.get(f"{scopeName}", "spSecret")
tenantId = dbutils.secrets.get(f"{scopeName}", "tenantId")
subscription_id = dbutils.secrets.get(f"{scopeName}", "subscriptionId")


hours = int(dbutils.widgets.get("hours"))
resource_group = dbutils.widgets.get("resource_group")
endpointName = dbutils.widgets.get("endpointName")
app_insights_name = dbutils.widgets.get("app_insights_name")
PATH_TO_MLFLOW_EXPERIMENT = dbutils.widgets.get("PATH_TO_MLFLOW_EXPERIMENT")
DB_NAME = dbutils.widgets.get("DB_NAME")

# COMMAND ----------

import mlflow
from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract Data from MLFlow

# COMMAND ----------

expId = mlflow.get_experiment_by_name(PATH_TO_MLFLOW_EXPERIMENT).experiment_id

df = spark.read.format("mlflow-experiment").load(expId)

refined_df = df.select(col('run_id'), col("experiment_id"), explode(map_concat(col("metrics"), col("params"))), col('start_time'), col("end_time")) \
                .filter("key != 'model'") \
                .select("run_id", "experiment_id", "key", col("value").cast("float"), col('start_time'), col("end_time")) \
                .groupBy("run_id", "experiment_id", "start_time", "end_time") \
                .pivot("key") \
                .sum("value") \
                .withColumn("trainingDuration", col("end_time").cast("integer")-col("start_time").cast("integer")) # example of added column

# COMMAND ----------

# if the table does not exist, create new table and append transformed data
# if the table DOES exists, merge the data into the table
if tableExists:
  print("beginning merge")
  existingTable = DeltaTable.forName(spark, f"{DB_NAME}.experiment_data_bronze")
  existingTable.alias("s").merge(
    refined_df.alias("t"),
    "s.run_id = t.run_id") \
  .whenNotMatchedInsertAll() \
  .execute()
  dbutils.notebook.exit(json.dumps({
    "status":200,
    "message":f"Data merged successfully into {DB_NAME}.experiment_data_bronze"
  }))
else:
  print("creating new table")
  refined_df.write.saveAsTable(f"{DB_NAME}.experiment_data_bronze", format="delta", mode="overwrite")
  dbutils.notebook.exit(json.dumps({
    "status":200,
    "message":f"{DB_NAME}.experiment_data_bronze table created, and data successsfully written to the table."
  }))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract Data from Azure Application Insights

# COMMAND ----------

from AzureExtractors import AzureLogClient
from AzureExtractors import AppInsightsReader

token = AzureLogClient.getADToken(tenantId, clientId, clientSecret)["accessToken"]
response_sample = AppInsightsReader.get_response_sample_from_endpoint(token, subscription_id, resource_group, app_insights_name, endpointName, 5)
appInsightsDF = AppInsightsReader.toSparkDataFrame(spark, response_sample)

# COMMAND ----------

appInsightsDF_Filtered = appInsightsDF.filter(appInsightsDF.endpointName.contains(endpointName))

def getLatestRunId (pathToExperiment):
  import mlflow
  from pyspark.sql.functions import col
  expId = mlflow.get_experiment_by_name(pathToExperiment).experiment_id
  lastest_run_id = spark.read.format("mlflow-experiment").load(expId).orderBy(col("end_time").desc()).select("run_id").limit(1).collect()[0][0]
  return lastest_run_id

## add pathToExperiment, run_id, endpointName
appInsightsDF_Filtered = appInsightsDF_Filtered.withColumn("pathToExperiment", lit(PATH_TO_MLFLOW_EXPERIMENT)) \
                                                .withColumn("run_id", lit(getLatestRunId(PATH_TO_MLFLOW_EXPERIMENT))) \
                                                .withColumn("endpointName", lit(endpointName)) \
                                                .withColumn("deploymentTarget", lit("aci"))

# COMMAND ----------

tableExists = spark._jsparkSession.catalog().tableExists(DB_NAME, f"response_data_bronze")

if tableExists:
  print("beginning merge")
  
  existingTable = DeltaTable.forName(spark, f"{DB_NAME}.response_data_bronze")
  existingTable.alias("s").merge(
    appInsightsDF_Filtered.alias("t"),
    "s.requestID = t.requestID") \
  .whenNotMatchedInsertAll() \
  .execute()
  dbutils.notebook.exit(json.dumps({
    "status":200,
    "message":f"Data merged successfully into response_data_bronze"
  }))
else:
  print("creating new table")
  appInsightsDF_Filtered.write.saveAsTable(f"{DB_NAME}.response_data_bronze", format="delta", mode="overwrite")
  dbutils.notebook.exit(json.dumps({
    "status":200,
    "message":f"{DB_NAME}.response_data_bronze table created, and data successsfully written to the table."
  }))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bronze --> Silver for Application Insights Data

# COMMAND ----------

bronze_data = spark.read.table(f"{DB_NAME}.response_data_bronze")

columns = bronze_data.columns

bronze_data = bronze_data.select(col("timestamp").cast("timestamp"), *columns[1:])

def str_to_json(t):
  return json.loads(t)
jsonify = udf(str_to_json, StringType())

spark.udf.register('jsonify', jsonify)

columns_copy = columns.copy()
columns_copy.remove("response")

schema = StructType([
  StructField("columns", ArrayType(StringType()), True),
  StructField("index", ArrayType(IntegerType()), True),
  StructField("data", ArrayType(ArrayType(FloatType())), True)
])

bronze_data_processed = bronze_data.select(split(regexp_replace("response", '(\"\[|\]\")', ""), ",").cast("array<float>").alias("response"), *columns_copy) \
                                    .withColumn("processedInput", from_json(jsonify(col("inputData")), schema)) \
                                    .withColumn("input", col("processedInput.data")) \
                                    .withColumn("extractedColumns", col("processedInput.columns")) \
                                    .select(explode("input").alias("inputPart"), "*").withColumn("mappedInput", map_from_arrays(col("extractedColumns"), col("inputPart"))) \
                                    .select("timestamp", "pathToExperiment","model", "run_id", "requestId", "response", "mappedInput") \
                                    .groupBy(["timestamp", "pathToExperiment","model", "run_id", "requestId", "response"]).agg(collect_list("mappedInput").alias('input')) \
                                    .withColumn("mappedInputandPrediction", struct(col("response"), col("input")))

# COMMAND ----------

tableExists = spark._jsparkSession.catalog().tableExists(DB_NAME, f"{DB_NAME}.response_data_silver")

if tableExists:
  print("beginning merge")
  existingTable = DeltaTable.forName(spark, f"{DB_NAME}.response_data_silver")
  existingTable.alias("s").merge(
    bronze_data_processed.alias("t"),
    "s.requestId = t.requestId") \
  .whenNotMatchedInsertAll() \
  .execute()
  dbutils.notebook.exit(json.dumps({
    "status":200,
    "message":f"Data merged successfully into {DB_NAME}.response_data_silver"
  }))
else:
  print("creating new table")
  bronze_data_processed.write.saveAsTable(f"{DB_NAME}.response_data_silver", format="delta", mode="overwrite")
  dbutils.notebook.exit(json.dumps({
    "status":200,
    "message":f"{DB_NAME}.response_data_silver table created, and data successsfully written to the table."
  }))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract Data from Azure Metrics

# COMMAND ----------

from AzureExtractors import AzureMetricsReader

metrics_data = AzureMetricsReader.collectMetricsData(token, subscription_id, resource_group, endpointName, 5, ["CpuUsage", "MemoryUsage", "NetworkBytesReceivedPerSecond", "NetworkBytesTransmittedPerSecond"])
unionedMetricsDF = AzureMetricsReader.toSparkDataFrame(spark, metrics_data)

unioned_metrics_for_model = unionedMetricsDF.withColumn("date", col("timeStamp").cast("date")) \
                                            .withColumn("hour", hour("timeStamp")) \
                                            .groupBy("endpoint_name", "date", "hour", "metric").agg(avg("average").alias("value")) \
                                            .groupBy("endpoint_name", "date", "hour").pivot("metric").sum("value") \
                                            .withColumn("timeStamp", (unix_timestamp(col('date').cast("timestamp"))+(col("hour")*3600)).cast("timestamp")) \
                                            .withColumn("MemoryUsageMB", col("MemoryUsage")/1000000) \
                                            .withColumn("pathToExperiment", lit(PATH_TO_MLFLOW_EXPERIMENT)) \
                                            .withColumn("run_id", lit(getLatestRunId(PATH_TO_MLFLOW_EXPERIMENT))) \
                                            .withColumn("endpointName", lit(endpointName)) \
                                            .withColumn("deploymentTarget", lit("aci"))

# COMMAND ----------

tableExists = spark._jsparkSession.catalog().tableExists(DB_NAME, f'endpoint_metrics_bronze')

if tableExists:
  print("beginning merge")
  existingTable = DeltaTable.forName(spark, f"{DB_NAME}.endpoint_metrics_bronze")
  existingTable.alias("s").merge(
    unioned_metrics_for_model.alias("t"),
    "s.timeStamp = t.timeStamp") \
  .whenNotMatchedInsertAll() \
  .execute()
  dbutils.notebook.exit(json.dumps({
    "status":200,
    "message":f"Data merged successfully into endpoint_metrics_bronze"
  }))
else:
  print("creating new table")
  unioned_metrics_for_model.write.saveAsTable(f"{DB_NAME}.endpoint_metrics_bronze", format="delta", mode="overwrite")
  dbutils.notebook.exit(json.dumps({
    "status":200,
    "message":f"{DB_NAME}.endpoint_metrics_bronze table created, and data successsfully written to the table."
  }))
