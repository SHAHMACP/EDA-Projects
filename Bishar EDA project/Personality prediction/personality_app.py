from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open('knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Fetch values from form
        Time_spent_Alone = float(request.form['Time_spent_Alone'])
        Stage_fear = int(request.form['Stage_fear'])
        Social_event_attendance = float(request.form['Social_event_attendance'])
        Going_outside = float(request.form['Going_outside'])
        Drained_after_socializing = int(request.form['Drained_after_socializing'])
        Friends_circle_size = float(request.form['Friends_circle_size'])
        Post_frequency = float(request.form['Post_frequency'])

        # Prepare input
        features = np.array([[Time_spent_Alone, Stage_fear, Social_event_attendance,
                              Going_outside, Drained_after_socializing,
                              Friends_circle_size, Post_frequency]])

        prediction = model.predict(features)
        result = 'Extrovert' if prediction[0] == 1 else 'Introvert'

        return render_template('form.html', prediction_text=f'Predicted Personality: {result}')

    except Exception as e:
        return render_template('form.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)

