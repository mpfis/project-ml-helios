# Databricks notebook source
dbutils.widgets.text("scoringURI", "")
scoring_uri = dbutils.widgets.get("scoringURI")

# COMMAND ----------

import random
import time

sample_jsons = [ {
    "columns": [
        "Sensor1",
        "Sensor2",
        "Sensor3",
        "Sensor4"
    ],
    "data": [
        [random.randint(50,80)+random.random(), 
         random.randint(11000,21000)+random.random(),
         random.randint(50,140)+random.random(),	
         random.randint(50,80)+random.random()]
    ]
} for i in range(0, 10000)]


anomalous_jsons = [ {
    "columns": [
        "Sensor1",
        "Sensor2",
        "Sensor3",
        "Sensor4"
    ],
    "data": [
        [random.randint(90,110)+random.random(), 
         random.randint(21500,23000)+random.random(),
         random.randint(145,165)+random.random(),	
         random.randint(87,107)+random.random()]
    ]
} for i in range(0, 100)]



import requests
import json

# Function for calling the API
def service_query(input_data, uri):
  response = requests.post(
              url=uri, data=json.dumps(input_data),
              headers={"Content-type": "application/json"})
  prediction = response.text
  print(prediction)
  return prediction

# COMMAND ----------

# API Call
for sample_json in sample_jsons:
  send_anomaly = True if random.randint(1, 100) > 98 else False
  if send_anomaly:
    service_query(anomalous_jsons[random.randint(0,99)], scoring_uri)
    print("anomaly sent")
  service_query(sample_json, scoring_uri)
  time.sleep(random.randint(3,10))

# COMMAND ----------


