import requests
import json
import logging
import datetime
import os
import sys
import pandas as pd
from delta.tables import *

def get_response_sample_from_endpoint (token, subscription_id, resource_group, app_insights_name, endpointName, hours):
  url = (f"https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/"+
         f"Microsoft.Insights/components/{app_insights_name}/api/query?api-version=2014-12-01-preview"+
         f"&query=traces%20|where%20customDimensions%20contains%20'input'%20|"+
         f"%20where%20customDimensions.['Service%20Name']%20==%20'{endpointName}'|%20where%20timestamp%20>%20ago({hours}h)")

  headers = {
    'Authorization': f'Bearer {token}',
    'Content-Type': 'application/json'
  }

  response = requests.request("GET", url, headers=headers)
  
  response_data = json.loads(response.content)['Tables'][0]
  return response_data

def extractRequiredAppInsightsData (row):
  return [row[0], json.loads(row[4])["Workspace Name"], json.loads(row[4])["Service Name"], json.loads(row[4])["Container Id"], 
          json.loads(row[4])["Prediction"], json.loads(row[4])["Request Id"], json.loads(row[4])["Models"], json.loads(row[4])["Input"], row[-5]]

def toList (response_data):
    return [extractRequiredAppInsightsData(row) for row in response_data['Rows']]


def toSparkDataFrame (response_data):
    rows = [extractRequiredAppInsightsData(row) for row in response_data['Rows']]
    appInsightsDF = spark.createDataFrame(rows, ["timestamp", "workspaceName", "endpointName", "containerId", "response", "requestId", "model", "inputData", "mlWorkspace"])
    return appInsightsDF