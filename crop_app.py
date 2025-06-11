from flask import Flask, render_template, request
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
best_model = joblib.load('crop_recommendation_model.pkl')  # Load the model
ms = joblib.load('scaler.pkl')  # Load the scaler

# Prediction function
def recommendation(N_SOIL, P_SOIL, K_SOIL, TEMPERATURE, HUMIDITY, ph, RAINFALL):
    features = pd.DataFrame([[N_SOIL, P_SOIL, K_SOIL, TEMPERATURE, HUMIDITY, ph, RAINFALL]],
                            columns=['N_SOIL', 'P_SOIL', 'K_SOIL', 'TEMPERATURE', 'HUMIDITY', 'ph', 'RAINFALL'])
    features_scaled = ms.transform(features)  # Scale features using the loaded scaler
    prediction = best_model.predict(features_scaled)  # Get the prediction
    return prediction[0]

# Route for homepage (input form)
@app.route('/')
def home():
    return render_template('landingPage.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user inputs
        N_SOIL = float(request.form['N_SOIL'])
        P_SOIL = float(request.form['P_SOIL'])
        K_SOIL = float(request.form['K_SOIL'])
        TEMPERATURE = float(request.form['TEMPERATURE'])
        HUMIDITY = float(request.form['HUMIDITY'])
        ph = float(request.form['ph'])
        RAINFALL = float(request.form['RAINFALL'])

        prediction = recommendation(N_SOIL, P_SOIL, K_SOIL, TEMPERATURE, HUMIDITY, ph, RAINFALL)

        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
            7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
            12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
            17: "MungBean", 18: "MothBeans", 19: "PigeonPeas", 20: "KidneyBeans",
            21: "ChickPea", 22: "Coffee"
        }

        recommended_crop = crop_dict.get(prediction, "No recommendation available")
        return render_template('predict.html', crop=recommended_crop)

@app.route('/predictcrop')
def predictcrop():
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
