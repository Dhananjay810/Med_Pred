from flask import Flask, request, jsonify
import pickle
import os
import numpy as np
from flask_cors import CORS

# Initialize app
app = Flask(__name__)
CORS(app)  # Enable CORS (important for Android/API calls)

# Load model safely
try:
    with open('pipe_lr.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)
    model = None


@app.route('/')
def home():
    return "Disease Prediction API Running 🚀"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if model loaded
        if model is None:
            return jsonify({'error': 'Model not loaded properly'})

        # Get JSON data
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data received'})

        if 'features' not in data:
            return jsonify({'error': 'Missing "features" key'})

        features = data['features']

        # Validate length
        if len(features) != 377:
            return jsonify({
                'error': f'Expected 377 features, got {len(features)}'
            })

        # Convert to integers safely
        try:
            features = [int(x) for x in features]
        except:
            return jsonify({'error': 'Features must be integers (0 or 1)'})

        # Convert to numpy
        input_data = np.array(features).reshape(1, -1)

        # Predict
        prediction = model.predict(input_data)

        return jsonify({
            'prediction': str(prediction[0]),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        })


# Run locally (Render will use gunicorn instead)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)