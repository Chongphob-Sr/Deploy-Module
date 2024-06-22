import os
from flask import Flask, jsonify, request, render_template
import joblib

app = Flask(__name__)

# Get the absolute path of the directory this script is in
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load your trained model
model_path = os.path.join(script_dir, 'trained_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    year = int(data['Year'])
    electric_vehicles = int(data['Electric_Vehicles'])
    
    # Perform prediction
    prediction = model.predict([[year, electric_vehicles]])
    
    # Return JSON response
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
