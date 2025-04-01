import pickle

from flask import Flask, request, jsonify

with open('models/lin_reg.bin', 'rb') as f_in:
    (dv, model, ls, lrid) = pickle.load(f_in)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = f'{str(ride['PULocationID'])}_{str(ride['DOLocationID'])}'
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    X = dv.transform(features)
    predictions = model.predict(X)
    return predictions

app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = round(predict(features)[0], 2)

    result = {'duration': pred}

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)