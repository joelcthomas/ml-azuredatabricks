# Databricks notebook source
# MAGIC %md
# MAGIC # An End to End ML Demo with Azure Databricks and MLflow integrated with Azure ML
# MAGIC ### Read Data, Build ML Model (Spark ML & Scikit learn), Track with MLflow, Compare Models, Model Registry, Deploy to production as batch with UDF & as REST endpoint with AML

# COMMAND ----------

# MAGIC %md
# MAGIC ### ML Objective:
# MAGIC Here we use a timeseries data from 5 sensors. Goal is to create a ML model that can predict Sensor 5 value based on other sensors

# COMMAND ----------

# MAGIC %md
# MAGIC ###Import Training Data
# MAGIC <img src="https://mcg1stanstor00.blob.core.windows.net/images/demos/Ignite/delta.jpg" alt="Delta" width="600">
# MAGIC </br></br>
# MAGIC The training data for this notebook is simply some time series data from devices that includes a collection of sensor readings.  
# MAGIC The data is stored in the Delta Lake format.  The data can be downloaded in CSV [here](https://mcg1stanstor00.blob.core.windows.net/publicdata/sensors/sensordata.csv).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initial Setup

# COMMAND ----------

#Install required modules
dbutils.library.installPyPI("azureml-mlflow")
dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
import mlflow
import mlflow.spark
import mlflow.sklearn
import mlflow.azureml
import azureml
import azureml.core
from azureml.core import Workspace

# COMMAND ----------

# MAGIC %md
# MAGIC ### Review Data

# COMMAND ----------

# Here data is already in Delta lake and registered as table within Databricks
# To simulate, download the data mentioned above, use the 'Data' tab on the left sidebar to upload and set as table.
dataDf = spark.table("sensor").where(col('Device') == 'Device001')
display(dataDf)

# COMMAND ----------

# MAGIC %md
# MAGIC #Experiment Tracking and Model Deployment 
# MAGIC ##with MLFlow and Azure Machine Learning
# MAGIC <img src="https://raw.githubusercontent.com/iheartdatascience/ignite2020/master/aml_adb.jpg" alt="Better Together" width="800">
# MAGIC </br></br>
# MAGIC This notebook walks through a basic Machine Learning example. Training runs will be logged to Azure Machine Learning using MLFlow's open-source APIs.  </br> A resulting model from one of the models will then be deployed using MLFlow APIs as a) a Spark Pandas UDF for batch scoring and b) a web service in Azure Machine Learning

# COMMAND ----------

# MAGIC %md
# MAGIC ##Basic Setup
# MAGIC <img src="https://raw.githubusercontent.com/iheartdatascience/ignite2020/master/notebookimage1.JPG" alt="Basic Setup" width="600">
# MAGIC </br></br>
# MAGIC 
# MAGIC Basic setup requires that the Databricks Workspace is linked with the AML workspace

# COMMAND ----------

# MAGIC %md
# MAGIC ##Experiment Tracking with MLFlow and AML
# MAGIC <img src="https://raw.githubusercontent.com/iheartdatascience/ignite2020/master/experiment.jpg" alt="Experiment Tracking" width="750">
# MAGIC </br>
# MAGIC MLFlow logging APIs will be used to log training experiments, metrics, and artifacts to AML.

# COMMAND ----------

#Set MLFlow Experiment
experimentName = "/Users/joel.thomas@databricks.com/ml101/ML_with_ADB_and_AML"
mlflow.set_experiment(experimentName)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ML Model with Spark ML
# MAGIC <img src="https://mcg1stanstor00.blob.core.windows.net/images/demos/Ignite/spark.jpg" alt="Spark" width="150">

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

# Split the data into training and test sets (30% held out for testing)
(train_data, test_data) = dataDf.randomSplit([0.7, 0.3])

# COMMAND ----------

# Incorporate all input fields as vector for regression pipeline
assembler = VectorAssembler(
    inputCols=["Sensor1", "Sensor2", "Sensor3", "Sensor4"],
    outputCol="features")

# COMMAND ----------

def regresionModel(stages, params, train, test):
  pipeline = Pipeline(stages=stages)
  
  with mlflow.start_run(run_name="Sensor Regression") as ml_run:
    for k,v in params.items():
      mlflow.log_param(k, v)

    model = pipeline.fit(train)
    predictions = model.transform(test)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol="Sensor5", predictionCol="prediction", metricName="mse")
    mse = evaluator.evaluate(predictions)

    evaluator = RegressionEvaluator(
        labelCol="Sensor5", predictionCol="prediction", metricName="r2")
    r2 = evaluator.evaluate(predictions)

    #Log MLFlow Metrics and Model
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.spark.log_model(model, "model")

    print("Documented with MLflow Run id %s" % ml_run.info.run_uuid)
  
  return mse, r2, ml_run.info

# COMMAND ----------

numTreesList = [10, 25]
maxDepthList = [5, 10]
for numTrees, maxDepth in [(numTrees,maxDepth) for numTrees in numTreesList for maxDepth in maxDepthList]:
  params = {"numTrees":numTrees, "maxDepth":maxDepth, "model": "Radom Forest Regressor - SparkML"}
  rf = RandomForestRegressor(featuresCol="features", labelCol="Sensor5", numTrees=numTrees, maxDepth=maxDepth)
  mse, r2, ml_run_info = regresionModel([assembler, rf], params, train_data, test_data)
  print("Trees: %s, Depth: %s, MSE: %s, R2: %s\n" % (numTrees, maxDepth, mse, r2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### ML Model with Scikit Learn
# MAGIC <img src="https://mcg1stanstor00.blob.core.windows.net/images/demos/Ignite/skl.jpg" alt="SciKit Learn" width="150">

# COMMAND ----------

import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

#Setup Test/Train datasets
data = dataDf.toPandas()

x = data.drop(["Device", "Time", "Sensor5"], axis=1)
y = data[["Sensor5"]]
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.20, random_state=30)

#Train Models
device = "Device001"

resultsPdf = pd.DataFrame()
for numTrees, maxDepth in [(numTrees,maxDepth) for numTrees in numTreesList for maxDepth in maxDepthList]:
  with mlflow.start_run(run_name="Sensor Regression"):
    
    mlflow.log_param("maxDepth", maxDepth)
    mlflow.log_param("numTrees", numTrees)
    mlflow.log_param("model", "Radom Forest Regressor - scikit")
    
    # Fit, train, and score the model
    model = RandomForestRegressor(max_depth = maxDepth, n_estimators = numTrees)
    model.fit(train_x, train_y)
    preds = model.predict(test_x)

    # Get Metrics
    mse = mean_squared_error(test_y, preds)
    r2 = r2_score(test_y, preds)

    # Log Metrics and Model
    mlflow.log_metric('mse', mse)
    mlflow.log_metric('r2', r2)
    mlflow.sklearn.log_model(model, "model")

    # Build Metrics Table
    results = [[device, maxDepth, numTrees, mse, r2]]
    runResultsPdf = pd.DataFrame(results, columns =['Device', 'MaxDepth', 'NumTrees', 'MSE', 'r2'])
    resultsPdf = resultsPdf.append(runResultsPdf)

    last_run_id = mlflow.active_run().info.run_id
  
display(resultsPdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Deployment
# MAGIC <img src="https://raw.githubusercontent.com/iheartdatascience/ignite2020/master/model_deployment.jpg" alt="Model Deployment" width="800">
# MAGIC </br></br>
# MAGIC Using MLFlow APIs, models can be deployed to AML and turned into web services, or they can be deployed as MLFlow model objects 
# MAGIC </br>and used in streaming or batch pipelines as Python functions or Pandas UDFs.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy Model for Batch Scoring
# MAGIC <img src="https://mcg1stanstor00.blob.core.windows.net/images/demos/Ignite/deploylake.jpg" alt="Model Deployment" width="800">
# MAGIC </br></br>
# MAGIC Using MLFlow APIs, the Scikit Learn MLFlow Model will be exported out of AML and put in the Data Lake where it can be more widely accessed.

# COMMAND ----------

model_uri = "runs:/"+last_run_id+"/model"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Use Apache Spark for Batch Scoring
# MAGIC <img src="https://raw.githubusercontent.com/iheartdatascience/ignite2020/master/batch_scoring.jpg" alt="Model Deployment" width="800">
# MAGIC </br></br>
# MAGIC The MLFlow model will be loaded and used as a Spark Pandas UDF to score new data.

# COMMAND ----------

from pyspark.sql.types import ArrayType, FloatType

#Create a Spark UDF for the MLFlow model
pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_uri)

#Load Scoring Data into Spark Dataframe
scoreDf = spark.table("sensor").where(col('Device') == 'Device100')

#Make Prediction
preds = (scoreDf
           .withColumn('Sensor5-prediction', pyfunc_udf('Sensor1', 'Sensor2', 'Sensor3', 'Sensor4'))
        )
display(preds)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy Model as a Web Service in AML
# MAGIC <img src="https://mcg1stanstor00.blob.core.windows.net/images/demos/Ignite/deploywebservice.jpg" alt="Model Deployment" width="800">
# MAGIC </br></br>
# MAGIC The MLFlow model will conainerized and deployed as a web service with AML and Azure Container Instances

# COMMAND ----------

# workspace_name = "<WORKSPACE_NAME>"
# workspace_location="<WORKSPACE_LOCATION>"
# resource_group = "<RESOURCE_GROUP>"
# subscription_id = "<SUBSCRIPTION_ID>"

workspace = Workspace.create(name = workspace_name,
                             subscription_id = subscription_id,
                             resource_group = resource_group,
                             location = workspace_location,
                             exist_ok=True)

# COMMAND ----------

experimentName = "ml101-webinar"
azure_service, azure_model = mlflow.azureml.deploy(model_uri=model_uri,
                                                   service_name=experimentName + "-service",
                                                   workspace=workspace,
                                                   synchronous=True)

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Score Using Web Service URI

# COMMAND ----------

# Create input data for the API
sample_json = {
    "columns": [
        "Sensor1",
        "Sensor2",
        "Sensor3",
        "Sensor4"
    ],
    "data": [
        [65.7845, 16613.676, 101.69767,	60.329124]
    ]
}

print(sample_json)

# COMMAND ----------

##Get the Web Service URI 
uri = azure_service.scoring_uri

# COMMAND ----------

import requests
import json

# Function for calling the API
def service_query(input_data):
  response = requests.post(
              url=uri, data=json.dumps(input_data),
              headers={"Content-type": "application/json"})
  prediction = response.text
  print(prediction)
  return prediction

# API Call
service_query(sample_json)

# COMMAND ----------


