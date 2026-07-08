**🖼️ AI Image Quality Checker**

A CNN-based binary image classifier that predicts whether an uploaded image is Good or Low quality, with a confidence-scored "uncertain" zone for borderline cases — built with TensorFlow/Keras and served through a Gradio interface.

---


**💡 Why This Matters**

Manually reviewing images for quality (blur, poor lighting, low resolution) doesn't scale — whether it's a content platform screening uploads or an e-commerce site checking product photos. This project demonstrates a lightweight CNN trained from scratch to make that judgment automatically, with data augmentation used to get the most out of a small labeled dataset.

Input: any uploaded image → Output: Good / Medium (uncertain) / Low quality with a confidence score.

---


**⚙️ Key Features**


🧠 Custom CNN (Conv2D → MaxPooling → Dense → Dropout) trained from scratch in TensorFlow/Keras
🔄 Data augmentation (rotation, shift, zoom, horizontal flip) to make the most of a small training set
📊 Confidence-scored prediction rather than a flat yes/no label
🎨 Simple drag-and-drop Gradio interface for instant testing
⚡ Fast local inference, no GPU required

----



**🧠 How It Works**


. User uploads an image through the Gradio interface

. Image is resized to 224×224 and normalized

. The CNN outputs a single confidence score between 0 and 1

. The score is mapped to a result:


> 0.7 → Good Quality
0.4–0.7 → Medium / Uncertain (the model isn't confident either way)
< 0.4 → Low Quality

----


## Demo

Demo Video:
https://github.com/Allure815/image_quality_checker/blob/main/Demo-Image%20Classifier.mp4

Screenshot:
https://github.com/Allure815/image_quality_checker/blob/main/img-ss.png

---


## Tech Stack

-Modeling: TensorFlow / Keras (custom CNN)

-Interface: Gradio

-Image Processing: Pillow, NumPy

-Language: Python


---


**📌 Current Status: Prototype**

This is a binary classifier under the hood — trained on two labeled classes, Good and Bad, using a small dataset (~20 images) with augmentation to help generalization. The "Medium Quality" result isn't a third trained class; it's a confidence band on the binary score, flagging predictions the model isn't sure about rather than a category it learned to recognize. Framed that way, it's an honest and useful signal (uncertain predictions genuinely deserve a second look) — it's just not the same as true 3-class classification, which is the next milestone below.


----


## Project Structure

image_quality_checker
model/
  image_quality_model.h5
app.py
utils.py
requirements.txt
README.md

---



**▶️ Run It Locally**

bash# Clone
git clone https://github.com/Allure815/image_quality_checker.git
cd image_quality_checker

**Create and activate a virtual environment**
python -m venv venv
venv\Scripts\activate      # Windows
**source** venv/bin/activate  # macOS/Linux

**Install dependencies**
pip install -r requirements.txt

**Run the app**
python app.py


Then open http://127.0.0.1:7860 and upload an image to test the prediction.

---



**🔭 What's Next**

Collect and label a genuine "Medium" quality class to move from a confidence-band heuristic to true 3-class classification
Expand the training dataset significantly beyond the current ~20 images for better generalization
Add explainability (e.g. Grad-CAM) so predictions show why an image was flagged as low quality
Deploy the app online (Hugging Face Spaces or similar) for live demoing without a local setup

---


## Model Behavior

The model outputs a prediction score between 0 and 1.

Score greater than 0.7 → Good Quality
Score between 0.4 and 0.7 → Medium Quality
Score less than 0.4 → Low Quality


----

**👤 Author**

Heeral — https://github.com/Allure815



