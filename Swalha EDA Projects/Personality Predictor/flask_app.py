from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained KNN model
with open('knn_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
        # Get and map "Yes"/"No" values to 1/0
        stage_fear = 1 if request.form['Stage_fear'] == 'Yes' else 0
        drained_after_socializing = 1 if request.form['Drained_after_socializing'] == 'Yes' else 0

        # Get other inputs
        time_spent_alone = float(request.form['Time_spent_Alone'])
        social_event_attendance = float(request.form['Social_event_attendance'])
        going_outside = float(request.form['Going_outside'])
        friends_circle_size = float(request.form['Friends_circle_size'])
        post_frequency = float(request.form['Post_frequency'])

        # Combine all features
        final_features = np.array([[time_spent_alone, stage_fear, social_event_attendance,
                                    going_outside, drained_after_socializing,
                                    friends_circle_size, post_frequency]])

        # Predict
        prediction = model.predict(final_features)
        result = 'Extrovert' if prediction[0] == 1 else 'Introvert'

        return render_template('index.html', prediction_text=f'Predicted Personality: {result}')
if __name__ == '__main__':
    app.run(debug=True)