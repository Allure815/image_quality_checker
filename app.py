import gradio as gr
from PIL import Image
from utils import predict_quality

# Function to use in Gradio
def classify_image(image):
    return predict_quality(image)

# Create Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Quality Checker",
    description="Upload an image to check if it's good quality or low quality."
)

# Launch the app
iface.launch()
