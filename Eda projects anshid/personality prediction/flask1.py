from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model (make sure 'knn_model.pkl' exists)
with open('knn_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Map form values
        time_spent_alone = float(request.form['Time_spent_Alone'])
        stage_fear = 1 if request.form['Stage_fear'] == 'Yes' else 0
        social_event_attendance = float(request.form['Social_event_attendance'])
        going_outside = float(request.form['Going_outside'])
        drained_after_socializing = 1 if request.form['Drained_after_socializing'] == 'Yes' else 0
        friends_circle_size = float(request.form['Friends_circle_size'])
        post_frequency = float(request.form['Post_frequency'])

        # Create feature array
        features = np.array([[time_spent_alone, stage_fear, social_event_attendance,
                              going_outside, drained_after_socializing,
                              friends_circle_size, post_frequency]])

        # Get prediction
        prediction = model.predict(features)
        result = 'Extrovert' if prediction[0] == 1 else 'Introvert'

        return render_template('index.html', prediction_text=f'Predicted Personality: {result}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)


