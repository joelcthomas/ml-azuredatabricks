# Machine Learning with Azure Databricks
Easy to get started collection of Machine Learning Examples in Azure Databricks

## Azure Databricks Reference Architecture - Machine Learning & Advanced Analytics

<img src="https://joelcthomas.github.io/ml-azuredatabricks/img/azure_databricks_reference_architecture.png" width="1300">

## Key Benefits:
- End to end integration from data access (ADLS, SQL DW, EventHub, Kafka, etc.), data prep, feature engineering, model building in single node or distributed, MLops with MLflow, integration with AzureML, Synapse, & other Azure services.
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

- PyTorch on a single node with MNIST dataset
- PyTorch, distributed with Horovod with MNIST dataset
- Using MLflow to track hyperparameters, metrics, log models/artifacts in AzureML
- Using MLflow to deploy a scoing server (REST endpoint) with ACI  

adding soon:
- Single node scikit-learn to distributed hyperparamter search using Hyperopt 
- Single node pandas to distributed using Koalas
- PandasUDF to distribute models over different subsets of data or hyperparameters
- Using databricks automl-toolkit in Azure Databricks
- Using automl from AzureML in Azure Databricks

## Run:
To reproduce examples provided here, please import attached `ml-azuredatabricks.dbc` file to databricks workspace.

[Instructions on how to import notebooks in databricks](https://docs.microsoft.com/en-us/azure/databricks/notebooks/notebooks-manage#--import-a-notebook)


For more information on using Azure Databricks  
https://docs.microsoft.com/en-us/azure/azure-databricks/