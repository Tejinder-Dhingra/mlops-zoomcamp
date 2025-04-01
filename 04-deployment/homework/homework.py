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

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

if __name__ == '__main__':
    args = parse_args()
    if args.year: year = int(args.year)
    if args.month: month = int(args.month)

    input_dir = 'input'
    trip_category = 'yellow'

    python_version = platform.python_version()
    filename = f'{trip_category}_tripdata_{year}-{month:02d}.parquet'
    input_file_url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}'

    # ## Training model

    if not os.path.exists(input_dir): 
        os.makedirs(input_dir)
        print(f'Downloading {filename}...')
        input_file = wget.download(input_file_url, out=input_dir)
    else:
        glob_str = f'{input_dir}/*.parquet'
        input_file = glob(glob_str)

    df = read_data(input_file)
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    df_result = pd.DataFrame()
    df_result['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result['predicted_duration'] = y_pred

    print('Q5. Parametrizing script and Mean predicted duration')
    print(f'Parameters: year={year}, month={month}')
    print(f'Input file: {input_file}')
    print(f'Mean predicted duration = {y_pred.mean():.3f}')

