from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
with open("knn_model.pkl", "rb") as file:
    model = pickle.load(file)

# Flask app initialization
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("form.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get form input
    input_features = [
        float(request.form['time_spent_alone']),
        int(request.form['stage_fear']),
        float(request.form['social_event_attendance']),
        float(request.form['going_outside']),
        int(request.form['drained_after_socializing']),
        float(request.form['friends_circle_size']),
        float(request.form['post_frequency'])
    ]

    # Convert to NumPy array and reshape
    features = np.array(input_features).reshape(1, -1)

    # Predict using the model
    prediction = model.predict(features)[0]

    # Convert label to readable text
    result = "Extrovert" if prediction == 1 else "Introvert"

    return render_template("form.html", prediction_text=f'Predicted Personality: {result}')

if __name__ == '__main__':
    app.run(debug=True)
