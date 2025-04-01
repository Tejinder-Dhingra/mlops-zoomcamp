import requests

ride = {
    "PULocationID" : 30,
    "DOLocationID" : 50,
    "trip_distance" : 20
}

url = 'http://127.0.0.1:9696/predict'
response = requests.post(url, json=ride)
print(f'Duration: {response.json()['duration']} mins')