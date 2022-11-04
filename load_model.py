# dependencies
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error
from read_data import *

# load model
with open('/Users/yothinpu/mlops-zoomcamp/models/lin_reg.bin', 'rb') as f_out:
    read_model = pickle.load(f_out)

#read and transform 

test_df = read_dataframe('green_tripdata_2021-03.parquet')
test_df['PU_DO'] = test_df['PULocationID'] + '_' + test_df['DOLocationID']
categorical = ['PU_DO']
numerical = ['trip_distance'] 
dv = read_model[0] # load from model pickle
test_dicts = test_df[categorical + numerical].to_dict(orient='records')
X_new = dv.transform(test_dicts)

# load and predict 
y_pred = read_model[1].predict(X_new)

# target for model scoring

test_df['duration'] = test_df.lpep_dropoff_datetime - test_df.lpep_pickup_datetime
test_df.duration = test_df.duration.apply(lambda td: td.total_seconds() / 60)
target = 'duration' # label
y_new_real = test_df[target].values
print(mean_squared_error(y_new_real, y_pred, squared=False))