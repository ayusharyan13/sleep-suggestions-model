from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
import google.generativeai as genai
import shap
import textwrap

app = Flask(__name__)

# -----------------------------------
# 1. Load the Trained Model
# -----------------------------------
MODEL_PATH = 'final_random_forest_model.pkl'

try:
    with open(MODEL_PATH, 'rb') as file:
        rf_model = pickle.load(file)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# -----------------------------------
# 2. Configure LLM (Google Generative AI)
# -----------------------------------
API_KEY = "AIzaSyB6g26IKzLUqNGyU4TXYmF56AN1HnXd71o" # Ensure this is set in your environment

if not API_KEY:
    raise ValueError("API key for Google Generative AI not found. Please set the 'GOOGLE_GENERATIVE_AI_API_KEY' environment variable.")

genai.configure(api_key=API_KEY)
gemini = genai.GenerativeModel('gemini-pro')

# -----------------------------------
# 3. Initialize SHAP Explainer
# -----------------------------------
# Note: SHAP requires access to the training data. Ensure you have X_train available.
# For this example, we'll assume that you have saved X_train during model training.
# If not, you need to retrain the model with SHAP in mind or save X_train during training.

# Load training data used for SHAP explainer
# Replace 'X_train.pkl' with your actual training data pickle file path
TRAIN_DATA_PATH = 'X_train.pkl'

try:
    with open(TRAIN_DATA_PATH, 'rb') as file:
        X_train = pickle.load(file)
    print("Training data loaded successfully for SHAP.")
except Exception as e:
    print(f"Error loading training data for SHAP: {e}")
    X_train = None

print(X_train.columns)

if X_train is not None:
    explainer = shap.Explainer(rf_model, X_train)
else:
    explainer = shap.Explainer(rf_model)

# -----------------------------------
# 4. Define Utility Functions
# -----------------------------------

def get_top_features(shap_explainer, input_features, top_n=2):
    """
    Extracts the top N features contributing to a specific prediction using SHAP.
    
    Parameters:
    - shap_explainer: SHAP explainer object.
    - input_features: Numpy array of input features.
    - top_n: Number of top features to extract.
    
    Returns:
    - List of tuples containing feature names and their SHAP values.
    """
    
    shap_values = shap_explainer(input_features)
    instance_shap = shap_values.values[0]
    feature_names = ['hours_slept', 'sleep_conversation_interaction', 'Walking_duration',
       'Noise_duration']
    print(input_features, feature_names)
    feature_shap_pairs = list(zip(feature_names, instance_shap))
    feature_shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return feature_shap_pairs[:top_n]

def generate_suggestions(top_features, forecast):
    """
    Generates personalized suggestions using Google's Generative AI based on top contributing features.
    
    Parameters:
    - top_features: List of tuples containing feature names and their SHAP values.
    - forecast: Integer representing the sleep quality forecast category (1 to 4).
    
    Returns:
    - String containing personalized suggestions.
    """
    # Prepare formatted feature contributions
    features_info = "\n".join([f"- **{feature}**: {shap_val:.2f}" for feature, shap_val in top_features])
    
    # Map forecast to descriptive string
    forecast_str = ["Very Good Sleep Quality", "Fairly Good Sleep Quality", 
                    "Fairly Bad Sleep Quality", "Very Bad Sleep Quality"][forecast - 1]
    
    prompt = f"""
    The predicted sleep quality is **{forecast_str}** based on the following factors:
    {features_info}
    
    Please provide personalized suggestions and actionable insights to help improve the person's sleep quality.
    """
    
    # Generate content using Gemini
    response = gemini.generate_content(prompt)
    feedback = response.text.strip()
    
    return feedback

def to_markdown(text):
    """
    Converts text to Markdown format for better readability.
    """
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

# -----------------------------------
# 5. Define the Prediction Endpoint
# -----------------------------------
@app.route('/predict', methods=['POST'])
def predict_sleep_quality():
    try:
        # Extract input parameters from the request
        data = request.get_json()
        print(data)
        
        # Validate and extract required parameters
        required_params = ['hours_slept', 'sleep_conversation_interaction', 'walking_duration', 'noise_duration']
        for param in required_params:
            if param not in data:
                return jsonify({'error': f"Missing parameter: {param}"}), 400
        
        # Extract and convert input parameters
        hours_slept = float(data['hours_slept'])
        sleep_conversation_interaction = float(data['sleep_conversation_interaction'])
        walking_duration = float(data['walking_duration'])
        noise_duration = float(data['noise_duration'])
        
        # Prepare input array for prediction
        input_features = np.array([[hours_slept, sleep_conversation_interaction, walking_duration, noise_duration]])
        
        # Make prediction
        prediction = rf_model.predict(input_features)[0]
        print(prediction)
        
        # Map prediction to qualitative level
        if prediction < 1.5:
            quality = "Very Good Sleep Quality"
            forecast = 1
        elif prediction < 2.5:
            quality = "Fairly Good Sleep Quality"
            forecast = 2
        elif prediction < 3.5:
            quality = "Fairly Bad Sleep Quality"
            forecast = 3
        else:
            quality = "Very Bad Sleep Quality"
            forecast = 4
        
        print(quality, forecast)
        # Get top contributing features using SHAP
        top_features = get_top_features(explainer, input_features, top_n=2)
        print(top_features)
        
        # Generate personalized suggestions using LLM
        feedback = generate_suggestions(top_features, forecast)
        print(feedback)
        
        # Structure the response
        response = {
            'predicted_sleep_quality_score': round(prediction, 2),
            'qualitative_sleep_quality': quality,
            'feedback': feedback
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -----------------------------------
# 6. Define the Home Route
# -----------------------------------
@app.route('/', methods=['GET'])
def index():
    return 'Sleep Quality Prediction Server is Running!'

# -----------------------------------
# 7. Run the Flask Application
# -----------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=8082)
