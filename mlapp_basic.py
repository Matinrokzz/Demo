import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import mlflow.sklearn

import logging
import os
import fire
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
gcs_bucket = "gs://mlflow_warehouse/"
mlflow_user = "postgres"
mlflow_pass = "Makhn2144@@"
postgresql_databse = "postgres"
tracking_uri = f"postgresql://{mlflow_user}:{mlflow_pass}@127.0.0.1:5432/{postgresql_databse}"
mlflow.set_tracking_uri(tracking_uri)

experiment_name = "experiment_iris_model"
## check if the experiment already exists
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(name=experiment_name,artifact_location=gcs_bucket) 
experiment = mlflow.get_experiment_by_name(experiment_name)

input_data_path = "input_data_path.csv"
model_file ="iris_model.pkl"

with mlflow.start_run(experiment_id = experiment.experiment_id,run_name= f"run_{experiment_name}") :
     
    #-------Load data -----------#
    iris = pd.read_csv(input_data_path)
X = iris.drop("Species", axis=1)
y = iris.Species
    
    #-------Define model and parameters----------#

pca = PCA()
logistic = SGDClassifier(loss='log', penalty='l2', max_iter=200, tol=1e-3, random_state=0)
logistic.get_params()
param_grid = {
                'pca__n_components': [2],
                'logistic__alpha': np.logspace(-2, 1, 2),
            }
mlflow.log_params(param_grid)
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
    
    #--------Training ----------#
    
logging.info("beginning training")
search = GridSearchCV(pipe, param_grid, cv=2, return_train_score=False)
search.fit(X, y)
logging.info(f"Best parameter (CV score={search.best_score_}):")
        
best_param_renamed = {f'best_{param}': value for param, value in search.best_params_.items()}
mlflow.log_params(best_param_renamed)
mlflow.log_metric("best_score", search.best_score_)
    
    #--------Save best model ----------#

logging.info("saving best model")
dump(search.best_estimator_, model_file)
mlflow.log_artifact(model_file)
    #mlflow.sklearn.log_model(search.best_estimator_,"test_model")
    #mlflow.pyfunc.log_model(model, python_model=ModelWrapper()) 
mlflow.log_params({"model_file":model_file})

history_run = mlflow.search_runs(ViewType.ACTIVE_ONLY)
history_run

run_id = history_run.loc[history_run['metrics.best_score'].idxmax()]['run_id']
atf_uri = history_run.loc[history_run['metrics.best_score'].idxmin()]['artifact_uri']
model_name = history_run.loc[history_run['metrics.best_score'].idxmin()]['params.model_file']
model_uri = f"{atf_uri}/{model_name}"
model_uri