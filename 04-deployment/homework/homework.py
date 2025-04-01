#!/usr/bin/env python
# coding: utf-8

# # 04 - Deployment
import os
import wget
import pickle
import platform
import pandas as pd
import json 

from glob import glob
from argparse import ArgumentParser

month = 3
year = 2023
input_dir = 'input'
output_dir = 'output'
trip_category = 'yellow'

python_version = platform.python_version()
filename = f'{trip_category}_tripdata_{year}-{month:02d}.parquet'
input_file_url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}'
output_file = f'{output_dir}/{filename}'

categorical = ['PULocationID', 'DOLocationID']

# ## Q5. Parametrizing the script
def parse_args():
    parser = ArgumentParser(prog='deployment', usage='%(prog)s [options]')
    parser.add_argument('-y', '--year', help='Year')
    parser.add_argument('-m', '--month', help='Month')
    return parser.parse_args()

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

# ## Training model

if not os.path.exists(input_dir): 
    os.makedirs(input_dir)
    input_file = wget.download(input_file_url, out=input_dir)
else:
    glob_str = f'{input_dir}/*.parquet'
    input_file = glob(glob_str)

df = read_data(input_file)
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

# ## Q1. Standard deviation

print((f'Standard Deviation = {y_pred.std():.3f}'))

# ## Q2. Preparing the output

df_result = pd.DataFrame()
df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
df_result['predicted_duration'] = y_pred

if os.path.exists(output_file):
    os.remove(output_file)
else:
    if not os.path.exists(output_dir): os.makedirs(output_dir)
df_result.to_parquet(
    output_file,
    engine = 'pyarrow',
    compression = None,
    index = False,
)
print(f'{filename} - {(os.path.getsize(output_file)/1024**2):.2f}M')

with open('Pipfile.lock', 'r') as f:
    lock_data = json.load(f)
print(f'First hash of Scikit-learn dependency = "{lock_data['default']['scikit-learn']['hashes'][0]}"')

args = parse_args()
print(args)