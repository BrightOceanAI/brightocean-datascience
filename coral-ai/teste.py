from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

model = load_model('models/coral-ai.h5')

# Load and preprocess the image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescaling
    return img_array

# Function to make a prediction
def predict_image(model, img_path):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    return predictions

img_path = 'test/teste.jpg'
predictions = predict_image(model, img_path)

class_labels = ['Saudável', 'Branqueado']

predicted_class = class_labels[np.argmax(predictions)]
confidence = np.max(predictions)

print(f'Predicted Class: {predicted_class}, Confidence: {confidence}')

# Optionally, display the image
img = image.load_img(img_path)
plt.imshow(img)
plt.title(f'Predicted Class: {predicted_class}, Confidence: {confidence}')
plt.axis('off')
plt.show()