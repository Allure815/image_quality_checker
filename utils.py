import numpy as np
from PIL import Image
import tensorflow as tf

# Load the real trained model
model = tf.keras.models.load_model("model/image_quality_model.h5")

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_quality(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    score = prediction[0][0]
    
    if score > 0.5:
        return f"✅ Good Quality Image (Score: {score:.2f})"
    else:
        return f"❌ Low Quality Image (Score: {score:.2f})"


	 


