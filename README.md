# Machine Learning with Azure Databricks
Easy to get started collection of Machine Learning Examples in Azure Databricks  

Example Notebooks: [HTML format](https://joelcthomas.github.io/ml-azuredatabricks/), [Github](https://github.com/joelcthomas/ml-azuredatabricks)  

## Azure Databricks Reference Architecture - Machine Learning & Advanced Analytics

<img src="https://joelcthomas.github.io/ml-azuredatabricks/img/azure_databricks_reference_architecture.png" width="1300">

## Key Benefits:
- Built for enterprise with security, reliability, and scalability
- End to end integration from data access (ADLS, SQL DW, EventHub, Kafka, etc.), data prep, feature engineering, model building in single node or distributed, MLops with MLflow, integration with AzureML, Synapse, & other Azure services.
- Delta Lake to set the data foundation with higher data quality, reliability and performance for downstream ML & AI use cases 
- ML Runtime Optimizations
    - Reliable and secure distribution of open source ML frameworks
    - Packages and optimizes most common ML frameworks
    - Built-in optimization for distributed deep learning
    - Built-in AutoML and Experiment tracking
    - Customized environments using conda for reproducibility
- Distributed Machine Learning
    - Spark MLlib
    - Migrate Single Node to distributed with just a few lines of code changes:
        - Distributed hyperparameter search (Hyperopt, Gridsearch)
        - PandasUDF to distribute models over different subsets of data or hyperparameters
        - Koalas: Pandas DataFrame API on Spark
    - Distributed Deep Learning training with Horovod
- Use your own tools
    - Multiple languages in same Databricks notebooks (Python, R, Scala, SQL)
    - Databricks Connect: connect external tools with Azure databricks (IDEs, RStudio, Jupyter,...)

## Machine Learning & MLops Examples using Azure Databricks:
To review example notebooks below in HTML format: https://joelcthomas.github.io/ml-azuredatabricks/  
To reproduce in a notebook, see instructions below.

- [PyTorch on a single node with MNIST dataset](https://joelcthomas.github.io/ml-azuredatabricks/#PyTorch-SingleNode.html)
- [PyTorch, distributed with Horovod with MNIST dataset](https://joelcthomas.github.io/ml-azuredatabricks/#PyTorch-Horovod.html)
- [Using MLflow to track hyperparameters, metrics, log models/artifacts in AzureML](https://joelcthomas.github.io/ml-azuredatabricks/#PyTorch-SingleNode.html)
- [Using MLflow to deploy a scoing server (REST endpoint) with ACI](https://joelcthomas.github.io/ml-azuredatabricks/#PyTorch-SingleNode.html)  

adding soon:
- Single node scikit-learn to distributed hyperparamter search using Hyperopt 
- Single node pandas to distributed using Koalas
- PandasUDF to distribute models over different subsets of data or hyperparameters
- Using databricks automl-toolkit in Azure Databricks
- Using automl from AzureML in Azure Databricks

## MLflow
[Overview of MLflow and its features](./mlflow.md)

## How to run this example?
To reproduce examples provided here, please import `ml-azuredatabricks.dbc` file in git root directory to databricks workspace.

[Instructions on how to import notebooks in databricks](https://docs.microsoft.com/en-us/azure/databricks/notebooks/notebooks-manage#--import-a-notebook)

## Setup Cluster
Create a cluster - https://docs.microsoft.com/en-us/azure/databricks/clusters/create  
GPU enabled Clusters - https://docs.microsoft.com/en-us/azure/databricks/clusters/gpu  
Install a library/package - https://docs.microsoft.com/en-us/azure/databricks/libraries  
Machine Learning Runtime - https://docs.microsoft.com/en-us/azure/databricks/runtime/mlruntime  
To see list of already available package in each runtime - https://docs.microsoft.com/en-us/azure/databricks/release-notes/runtime/releases

## Additional Information
For more information on using Azure Databricks  
https://docs.microsoft.com/en-us/azure/azure-databricks/