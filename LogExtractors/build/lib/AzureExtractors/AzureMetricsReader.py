import requests
import json
import logging
import datetime
import os
import sys
import pandas as pd
from delta.tables import *
from functools import reduce



def getResourceName (token, subscription_id, resource_group, endpoint_name):
    url = f"https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group}/resources?$filter=resourceType eq 'Microsoft.ContainerInstance/containerGroups'&api-version=2020-06-01"
    payload={}
    headers = {
    'Authorization': f'Bearer {token}'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    resourceName = [resource["name"] for resource in response.json()['value'] if endpoint_name in resource["name"]][0]
    return resourceName


def getMetricsRetrievalURI (metricName, subscriptionId, resourceGroupName, isACIorAKS, resourceName, apiVersion="2018-01-01", hours=24):
  """
  
  """
  if isACIorAKS.lower() == "aci":
    resourceProviderNamespace = "Microsoft.ContainerInstance"
    resourceType = "containerGroups"
  elif isACIorAKS.lower() == "aks":
    resourceProviderNamespace = "Microsoft.ContainerService"
    resourceType = "managedClusters"
  else:
    print("Not supported")
    raise ValueError
  
  
  end = datetime.datetime.now()
  start = (end-datetime.timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
  end = end.strftime("%Y-%m-%dT%H:%M:%SZ")
  
  metricsURI = f"https://management.azure.com/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}"+ \
                f"/providers/{resourceProviderNamespace}/{resourceType}/{resourceName}"+ \
                f"/providers/microsoft.insights/metrics?api-version={apiVersion}&" + \
                f"metricnames={metricName}&timespan={start}/{end}"
  return metricsURI


def getMetricsData (token, metricsURI, endpointName, metricName):
  header = {
    "Content":"application/json",
    "Accept":"application/json",
    "Authorization":f"Bearer {token}"
  }
  request = requests.get(metricsURI, headers=header)
  jsons = json.dumps([json for json in json.loads(request.content)["value"][0]["timeseries"][0]["data"] if json.get("average") != None])
  df = pd.read_json(jsons)
  df["endpoint_name"] = endpointName
  df["metric"] = metricName
  return df


def collectMetricsData (token, subscription_id, resource_group, endpoint_name, hours, metrics):
    metricsDfs = []
    resourceName = getResourceName(token, subscription_id, resource_group, endpoint_name)
    for metric in metrics:
        metricsUri = getMetricsRetrievalURI(metric, subscription_id, resource_group, "aci", resourceName, hours=hours)
        metricsData = getMetricsData(token, metricsUri, endpoint_name, metric)
        metricsDfs.append(metricsData)
    return metricsDfs

def toSparkDataFrame (spark, metricsDfs):
    metricsDfsSpark = [spark.createDataFrame(df) for df in metricsDfs]
    return reduce(DataFrame.unionAll, metricsDfsSpark)