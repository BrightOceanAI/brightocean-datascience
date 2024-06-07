# App
import json
import logging
import werkzeug.exceptions
from flask_cors import CORS
from flask import Flask, request, jsonify

# Enviroment
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# API
import requests
from PIL import Image
from io import BytesIO

# Machine Learning
from tensorflow.keras.models import load_model
import tensorflow
import numpy as np


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.app_context().push()
logging.getLogger('flask_cors').level = logging.DEBUG

model = load_model('models/coral-ai.h5')


def load_and_preprocess_image(img):
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale pixel values to [0, 1]
    return img_array


def predict_image(model, image):
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    labels = ['Saudável', 'Branqueado']
    predicted_label = labels[predicted_class_index]
    confidence_score = prediction[0, predicted_class_index]
    return predicted_label, confidence_score


@app.route("/")
def home():
    return "Bem vindo a home de Machine Learning da Bright Ocean."


@app.route('/prediction', methods=['POST'])
def health_prediction():
    if request.method == 'POST':
        try:
            data = json.loads(request.data)
            link = data['link']
            response = requests.get(link, stream=True)
            response.raise_for_status()
            bytes = np.frombuffer(response.content, dtype='uint8')
            image = Image.open(BytesIO(bytes))
            image_array = load_and_preprocess_image(image)
            prediction = predict_image(model, image_array)

            return jsonify({'Saúde': f'{prediction[0]}'}), 200

        except Exception as error:
            return jsonify({'error': str(error)}), 406


# Error Handling
@app.errorhandler(werkzeug.exceptions.BadRequest)
def handleBadRequest(error):
    return jsonify({'error': 'Má requisição'}), 400


# Error Handling
@app.errorhandler(werkzeug.exceptions.MethodNotAllowed)
def handleMethodNotAllowed(error):
    return jsonify({'error': 'Método não permitido'}), 405


# Remove for deploy in Azure Web App Service
if __name__ == '__main__':
        app.run()