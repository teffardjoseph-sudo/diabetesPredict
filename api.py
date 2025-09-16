from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('model.joblib')  # Make sure model.joblib is in the same folder

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from POST request
        input_data = request.get_json()

        # Required input fields
        required_fields = ["Glucose", "Blood pressure", "Body mass index", "Age"]
        for field in required_fields:
            if field not in input_data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Extract and format the data for prediction
        data = np.array([
            input_data["Glucose"],
            input_data["Blood pressure"],
            input_data["Body mass index"],
            input_data["Age"]
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(data)[0]
        result = "No Risk" if prediction == 0 else "Risk"

        # Return the prediction
        return jsonify({
            "input": input_data,
            "prediction": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Host 0.0.0.0 to listen on all network interfaces (so ESP32 can connect)
    print("Starting Flask server...")
    app.run(host='192.168.0.204', port=5000)
