from google.cloud import storage
import pandas as pd
# create storage client
storage_client = storage.Client.from_service_account_json('/Users/matin-s_mac/Documents/Professional/Sirius/Data Science/Learning/Demand_forecasting/keys/demopurpose-314309-3167f3317055.json')
# get bucket with name
bucket = storage_client.get_bucket('demo_mlflow_warehouse')
# get bucket data as blob
test = bucket.get_blob('test.csv')
train = bucket.get_blob('train.csv')
with test.open("rt") as ts:
    df=pd.read_csv(ts)
    df.to_csv(r'/Users/matin-s_mac/Documents/Professional/Sirius/Data Science/Learning/Demand_forecasting/Forecasting Demo/updated/Forecasting Demo/data/test_out.csv', sep = ',', index=False)
with train.open("rt") as tr:
    df1=pd.read_csv(tr)
    df1.to_csv(r'/Users/matin-s_mac/Documents/Professional/Sirius/Data Science/Learning/Demand_forecasting/Forecasting Demo/updated/Forecasting Demo/data/train_out.csv', sep = ',', index=False)

import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings("ignore")

# ---------------Import and Preprocessing Data----------------
# Set location address to import, dataset name, all result file name 
local_add = '/Users/matin-s_mac/Documents/Professional/Sirius/Data Science/Learning/Demand_forecasting/Forecasting Demo/updated/Forecasting Demo'
data_folder = '/data'
result_folder = '/img'
model_folder = '/XGBoost'
train = '/train_out.csv'
test  = '/test_out.csv'
model_sc_xg = '/model_scores_XGBoost.p'
data_result_2017_predict = '/XG_2017_prediction.csv'
data_result_6m_predict = '/XG_6m_prediction.csv'
data_result_2017_18_predict = '/XG_2017_18_prediction.csv'
# Import dataset
df_train = pd.read_csv(local_add+data_folder+train)
df_test  = pd.read_csv(local_add+data_folder+test)
# Set variable date to timestamp
df_train.date = pd.to_datetime(df_train.date)
df_test.date  = pd.to_datetime(df_test.date)
# Add variable into the dataset
df_train['day_of_week'],  df_test['day_of_week']  = df_train.date.dt.dayofweek,  df_test.date.dt.dayofweek
df_train['month'],df_test['month'] = df_train.date.dt.month, df_test.date.dt.month
df_train['year'], df_test['year']  = df_train.date.dt.year,  df_test.date.dt.year

# ---------------Create and Split Data For Model----------------
# Make the time perioid for create train and test dataset
time_period = '2017-01-01'
# X_train
# Create X_train as 2013-01-01:2016-12-31 for train model
X_train = df_train.loc[df_train.date < time_period]
# Delete some unnecessary feature in X_train
X_train = X_train.drop(['date', 'sales'], axis=1)
# y_train
# Create y_train as target feature 2013-01-01:2016-12-31 for result
y_train = df_train.loc[df_train.date < time_period].sales
# X_test
# Create X_test as 2017-01-01:2017-12-31 for test model
X_test  = df_train.loc[df_train.date >= time_period]
# Delete some unnecessary feature in X_test
X_test  = X_test.drop(['date', 'sales'], axis=1)
# y_test
# Create y_test as target feature 2017-01-01:2017-12-31 for checking
y_test  = df_train.loc[df_train.date >= time_period].sales

# -----------------Model construct and train model------------------
# Model and 2017 prediction
model = XGBRegressor(n_estimators=1000, 
                     learning_rate=0.2, 
                     objective='reg:squarederror')
model.fit(X_train, y_train)
predictions_2017 = model.predict(X_test)

# -----Create freture for checking the prediction after train model------
# Create monthly sales date for plot with all raw dataset
monthly_data = df_train.copy()
monthly_data.date = monthly_data.date.apply(lambda x: str(x)[:7])
monthly_data = monthly_data.groupby('date')['sales'].sum().reset_index()
monthly_data.date = pd.to_datetime(monthly_data.date)
# Create monthly sales date for plot with 2017 prediction
monthly_data_2017 = df_train.loc[df_train.date >= time_period].date
monthly_data_2017 = pd.DataFrame(monthly_data_2017)
monthly_data_2017['sales'] = predictions_2017
monthly_data_2017.date = monthly_data_2017.date.apply(lambda x: str(x)[:7])
monthly_data_2017 = monthly_data_2017.groupby('date')['sales'].sum().reset_index()
monthly_data_2017.date = pd.to_datetime(monthly_data_2017.date)
monthly_data_2017 = pd.concat([monthly_data.loc[monthly_data.date == '2016-12-01'],
                               monthly_data_2017])
# Create only 2017 sales data and prediction sales data
result_predict_2017 = df_train.loc[df_train.date >= time_period]
result_predict_2017 = pd.DataFrame(result_predict_2017)
result_predict_2017['pre_sales'] = predictions_2017
result_predict_2017 = result_predict_2017.drop(['day_of_week','month','year'],axis=1)
# Save result 2017 prediction as a CSV file
result_predict_2017.to_csv(local_add+result_folder+model_folder+data_result_2017_predict)

# --------------------Checking Model Score----------------------
# Make the score model prediction of 2017 with RMSE, MAE, R2 and Precentage 
rmse = np.sqrt(mean_squared_error(monthly_data.sales.tail(12), monthly_data_2017.loc[monthly_data_2017.date >= time_period].sales))
mae = mean_absolute_error(monthly_data.sales.tail(12), monthly_data_2017.loc[monthly_data_2017.date >= time_period].sales)
r2 = r2_score(monthly_data.sales.tail(12), monthly_data_2017.loc[monthly_data_2017.date >= time_period].sales)
percentage_off = round(mae/monthly_data.sales.tail(12).mean()*100, 2)
percentage_actual = ((100/monthly_data.sales.tail(12).sum())*monthly_data_2017.loc[monthly_data_2017.date >= time_period].sales.sum())
# Save result of model all score as a pickle file (.p)
model_scores = {}
model_scores['XGBoost'] = [rmse, mae, r2, percentage_off, percentage_actual]
pickle.dump(model_scores, open(local_add+result_folder+model_sc_xg, 'wb'))

# ---------------Plot the Model prediction after train----------------
# Plot raw dataset monthly sales and compare with model prediction
fig,  ax  = plt.subplots(figsize=(12,3))
fig.patch.set_facecolor('w')
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
sns.lineplot(monthly_data['date'], monthly_data['sales'], ax=ax, color='mediumblue', label='Total Sales')
ax.set(xlabel = "Month",
       ylabel = "Sales",
       title = "Monthly Sales 2013-2017") 
plt.savefig(local_add+result_folder+model_folder+'/XGBoost_result1.png', bbox_inches='tight', dpi=100)

fig2, ax2 = plt.subplots(figsize=(12,3))
fig2.patch.set_facecolor('w')
ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
second = monthly_data.groupby(monthly_data.date.dt.year)['sales'].mean().reset_index()
second.date = pd.to_datetime(second.date, format='%Y')
sns.lineplot((second.date + datetime.timedelta(6*365/12)), second['sales'], ax=ax2, color='red', label='Mean Sales')   
ax2.set(xlabel = "Month",
       ylabel = "Sales",
       title = "Mean Monthly Sales 2013-2017")    
plt.savefig(local_add+result_folder+model_folder+'/XGBoost_result2.png', bbox_inches='tight', dpi=100)   

fig3, ax3 = plt.subplots(figsize=(12,3))
fig3.patch.set_facecolor('w')
ax3.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
sns.lineplot('date', 'sales', data=monthly_data, ax=ax3, color='mediumblue', label='Total Sales')
sns.lineplot((second.date + datetime.timedelta(6*365/12)), second['sales'], ax=ax3, color='red', label='Mean Sales')   
ax3.set(xlabel = "Month",
       ylabel = "Sales",
       title = "Monthly Sales With Mean 2013-2017") 
plt.savefig(local_add+result_folder+model_folder+'/XGBoost_result3.png', bbox_inches='tight', dpi=100)

fig4, ax4 = plt.subplots(figsize=(12,3))
fig4.patch.set_facecolor('w')
ax4.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
sns.lineplot(monthly_data.date, monthly_data.sales, ax=ax4, label='Original', color='mediumblue')
sns.lineplot(monthly_data_2017.date, monthly_data_2017.sales, ax=ax4, label='Predicted', color='Red')  
ax4.set(xlabel = "Month",
       ylabel = "Sales",
       title  = "XGboost Sales Forecasting Prediction")
plt.savefig(local_add+result_folder+model_folder+'/XGBoost_result4.png', bbox_inches='tight', dpi=100)

fig5, ax5 = plt.subplots(figsize=(12,3))
fig5.patch.set_facecolor('w')
ax5.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
sns.lineplot(monthly_data.date.tail(13), monthly_data.sales.tail(13), ax=ax5, label='Original', color='mediumblue')
sns.lineplot(monthly_data_2017.date, monthly_data_2017.sales, ax=ax5, label='Predicted', color='Red')  
ax5.set(xlabel = "Month",
       ylabel = "Sales",
       title  = "XGboost Sales Forecasting Prediction in 2017")
plt.savefig(local_add+result_folder+model_folder+'/XGBoost_result5.png', bbox_inches='tight', dpi=100)

# ---------Test model as a prediction next 6 months----------
# Create data test for predict in 6 months
test_6m = df_test.drop('date',axis=1)
# Prediction in future 6 months
predictions_6m = model.predict(test_6m)

# -----Create freture for checking the prediction after test model------
# Create monthly sales date fot plot with 6 months prediction
monthly_data_6m = df_test.date
monthly_data_6m = pd.DataFrame(monthly_data_6m)
monthly_data_6m['sales'] = predictions_6m
monthly_data_6m.date = monthly_data_6m.date.apply(lambda x: str(x)[:7])
monthly_data_6m = monthly_data_6m.groupby('date')['sales'].sum().reset_index()
monthly_data_6m.date = pd.to_datetime(monthly_data_6m.date)
temp_plt = pd.concat([monthly_data,monthly_data_6m[:1]])

# -----------Plot the Model prediction after test 6 months-----------
# Plot monthly sales of the prediction 6 months
fig, ax = plt.subplots(figsize=(12,3))
fig.patch.set_facecolor('w')
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
sns.lineplot(temp_plt.date, temp_plt.sales, ax=ax, label='Original', color='mediumblue')
sns.lineplot(monthly_data_6m.date, monthly_data_6m.sales, ax=ax, label='Predicted', color='Red')  
ax.set(xlabel = "Month",
       ylabel = "Sales",
       title  = "XGboost Sales Forecasting Prediction Next 6 Months")
plt.savefig(local_add+result_folder+model_folder+'/XGBoost_result6.png', bbox_inches='tight', dpi=100)

# -----------Make a result of 6 months prediction CSV file-----------
# Reconstruct result data
result_predict_6m = df_test.copy()
result_predict_6m = result_predict_6m.drop(['day_of_week','month','year'],axis=1)
result_predict_6m = pd.DataFrame(result_predict_6m)
result_predict_6m['pre_sales'] = predictions_6m
# Save result prediction as a CSV file
result_predict_6m.to_csv(local_add+result_folder+model_folder+data_result_6m_predict)

# -----------Make a result of 2017 and 2018-6 months prediction CSV file-----------
# Reconstruct result data
result_predict_2017_6m = pd.concat([result_predict_2017,result_predict_6m])
result_predict_2017_6m = result_predict_2017_6m.drop(['sales'],axis=1)
# Save result prediction as a CSV file
result_predict_2017_6m.to_csv(local_add+result_folder+model_folder+data_result_2017_18_predict)

# -----------Plot the prediction after test 2017 and 6 months-----------
# Reconstruct result data for plot into monthly
temp_2017_6m_pre = monthly_data_2017.loc[monthly_data_2017.date < time_period]
temp_2017_6m_pre = pd.DataFrame(temp_2017_6m_pre)
temp_2017_6m_pre = temp_2017_6m_pre.rename(columns = {'sales': 'pre_sales'}, inplace = False)
temp_2017_6m = result_predict_2017_6m.copy()
temp_2017_6m.date = temp_2017_6m.date.apply(lambda x: str(x)[:7])
temp_2017_6m = temp_2017_6m.groupby('date')['pre_sales'].sum().reset_index()
temp_2017_6m.date = pd.to_datetime(temp_2017_6m.date)
temp_2017_6m = pd.concat([temp_2017_6m_pre,temp_2017_6m]) 
# Plot monthly sales of the prediction 2017 and 6 months
fig, ax = plt.subplots(figsize=(12,3))
fig.patch.set_facecolor('w')
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
sns.lineplot(temp_plt.date, temp_plt.sales, ax=ax, label='Original', color='mediumblue')
sns.lineplot(temp_2017_6m.date, temp_2017_6m.pre_sales, ax=ax, label='Predicted Sales', color='Red')  
ax.set(xlabel = "Month",
       ylabel = "Sales",
       title  = "XGboost Sales Forecasting Prediction")
plt.savefig(local_add+result_folder+model_folder+'/XGBoost_result7.png', bbox_inches='tight', dpi=100)
    
# -----------Upload the file to GCP-----------    
pdf = pd.read_csv(local_add+result_folder+model_folder+data_result_2017_18_predict)
pdf.to_csv('gs://demo_mlflow_warehouse/prediction.csv')


# -----------MLFlow script--------------------

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from math import exp
import mlflow.xgboost
import numpy as np
import sys
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
from sklearn.linear_model import ElasticNet
from mlflow.utils.environment import _mlflow_conda_env
from urllib.parse import urlparse



search_space = {
  'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
  'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
  'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
  'objective': 'binary:logistic',
  'seed': 123, # Set a seed for deterministic training
}
 
def train_model(params):
  # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
  mlflow.xgboost.autolog()
  with mlflow.start_run(nested=True):
    train = xgb.DMatrix(data=X_train, label=y_train)
    test = xgb.DMatrix(data=X_test, label=y_test)
    # Pass in the test set so xgb can track an evaluation metric. XGBoost terminates training when the evaluation metric
    # is no longer improving.
    booster = xgb.train(params=params, dtrain=train, num_boost_round=1000,\
                        evals=[(test, "test")], early_stopping_rounds=50)
    predictions_test = booster.predict(test)
    auc_score = roc_auc_score(y_test, predictions_test)
    mlflow.log_metric('auc', auc_score)
 
    signature = infer_signature(X_train, booster.predict(train))
    mlflow.xgboost.log_model(booster, "model", signature=signature)
    
    # Set the loss to -1*auc_score so fmin maximizes the auc_score
    return {'status': STATUS_OK, 'loss': -1*auc_score, 'booster': booster.attributes()}
 
# Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep. 
# A reasonable value for parallelism is the square root of max_evals.
#spark_trials = SparkTrials(parallelism=10)
 
# Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
# run called "xgboost_models" .
#with mlflow.start_run(run_name='xgboost_models'):
#  best_params = fmin(
#    fn=train_model, 
#    space=search_space, 
#    algo=tpe.suggest, 
#    max_evals=96,
#    trials=spark_trials, 
#    rstate=np.random.RandomState(123)
#  )

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#def main():
#    rmse=rmse
#    mae=maer2=r2
#    return rmse, mae, r2

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    mlflow.set_tracking_uri("http://localhost:5000") 


with mlflow.start_run():
        lr = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)
        lr.fit(X_train, y_train)

        predicted_qualities = lr.predict(X_test)

        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

#        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

#        mlflow.log_param("alpha", alpha)
#        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="sales_prediction")
        else:
            mlflow.sklearn.log_model(lr, "model")

#from sklearn.externals import joblib
import joblib
import os
filename = os.path.join('./', 'final_model_XGBOOST.joblib')
joblib.dump(model,filename)




