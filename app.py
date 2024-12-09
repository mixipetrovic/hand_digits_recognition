from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import joblib

#Load the trained model
mlp=joblib.load('mlp_digit_classifier.pkl')

#Initialize the FLask app
app=Flask(__name__)

#Route to the main page
@app.route('/')
def home():
    return render_template('index.html') #Add html page to templates

# Route for predicting the digit
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if an image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No selected file'}), 400

        file = request.files['image']

        # If no filename is provided
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Ensure the file is an image
        if not file.content_type.startswith('image'):
            return jsonify({'error': 'Uploaded file is not an image'}), 400

        # Process the image
        img = Image.open(file.stream)
        img = img.convert('L')  # Convert to grayscale
        img = img.resize((8, 8), Image.Resampling.LANCZOS)  # Resize to 8x8 pixels
        
        img_data = np.array(img)  # Convert to numpy array
        img_data = img_data.reshape(1, -1)  # Flatten the image into a 1D array
        img_data = img_data / 16.0  # Normalize the data (match training dataset normalization)

        # Make the prediction
        prediction = mlp.predict(img_data)
        return jsonify({'predicted_digit': int(prediction[0])}), 200

    except Exception as e:
        # Log the exception for debugging
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)