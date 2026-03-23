import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.mixed_precision import Policy

# Fix old batch_shape issue
class CustomInputLayer(InputLayer):
    def __init__(self, **kwargs):
        if "batch_shape" in kwargs:
            kwargs["input_shape"] = kwargs.pop("batch_shape")[1:]
        super().__init__(**kwargs)

# Register custom objects
custom_objects = {
    "InputLayer": CustomInputLayer,
    "DTypePolicy": Policy
}

# Load trained model
model = load_model(
    "model/image_quality_model.h5",
    compile=False,
    custom_objects=custom_objects
)

def preprocess_image(image):
    image = image.convert("RGB")          # Ensure RGB
    image = image.resize((224, 224))      # Resize for model
    image = np.array(image) / 255.0       # Normalize
    image = np.expand_dims(image, axis=0) # Add batch dimension
    return image

def predict_quality(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    score = float(prediction[0][0])
    print("Prediction score:", score)

    if score > 0.7:
        return f"✅ Good Quality Image (Score: {score:.2f})"
    elif score > 0.4:
        return f"⚠️ Medium Quality Image (Score: {score:.2f})"
    else:
        return f"❌ Low Quality Image (Score: {score:.2f})"