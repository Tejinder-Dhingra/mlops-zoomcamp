import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg
import joblib


from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, ColumnQuantileMetric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	timestamp timestamp,
	prediction_drift float,
	fare_amount_quantile float
)
"""

reference_data = pd.read_parquet('data/reference.parquet')
with open('models/lin_reg.bin', 'rb') as f_in:
	model = joblib.load(f_in)

raw_data = pd.read_parquet('data/reference.parquet')

begin = datetime.datetime(2024, 3, 1, 0, 0)
num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
cat_features = ['PULocationID', 'DOLocationID']
column_mapping = ColumnMapping(
    target=None,
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features
)
report = Report(metrics=[
    ColumnDriftMetric(column_name='prediction'),
    ColumnQuantileMetric(column_name='fare_amount', quantile=0.5)
    ]
)

def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=postgres", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='evidently'")
		if len(res.fetchall()) == 0:
			conn.execute("create database evidently;")
		with psycopg.connect("host=localhost port=5432 dbname=evidently user=postgres password=postgres") as conn:
			conn.execute(create_table_statement)

def calculate_metrics_postgresql(curr, i):
	current_data = raw_data[(raw_data.lpep_pickup_datetime >= (begin + datetime.timedelta(i))) &
		(raw_data.lpep_pickup_datetime < (begin + datetime.timedelta(i + 1)))]
	if not current_data.empty:
		current_data['prediction'] = model.predict(current_data[num_features + cat_features].fillna(0))
		report.run(reference_data = reference_data, current_data = current_data,
            column_mapping=column_mapping)
		result = report.as_dict()
		prediction_drift = result['metrics'][0]['result']['drift_score']
		fare_amount_quantile = result['metrics'][1]['result']['current']['value']
		curr.execute(
            "insert into dummy_metrics(timestamp, prediction_drift, fare_amount_quantile) values (%s, %s, %s)",
            (begin + datetime.timedelta(i), prediction_drift, fare_amount_quantile)
        )

def batch_monitoring_backfill():
	prep_db()
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
	with psycopg.connect("host=localhost port=5432 dbname=evidently user=postgres password=postgres", autocommit=True) as conn:
		for i in range(0, 31):
			with conn.cursor() as curr:
				calculate_metrics_postgresql(curr, i)

			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("data sent")

if __name__ == '__main__':
	batch_monitoring_backfill()