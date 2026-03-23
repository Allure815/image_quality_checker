
---

# AI Image Quality Checker

AI Image Quality Checker is a simple web application that predicts whether an uploaded image is **Good Quality, Medium Quality, or Low Quality** using a trained deep learning model. The application provides a user-friendly interface where users can upload images and instantly receive quality predictions.

The project is built using **Python, TensorFlow, and Gradio**.

---

## Features

* Upload an image and check its quality instantly
* AI-based image quality prediction
* Three prediction categories:

  * Good Quality
  * Medium Quality
  * Low Quality
* Simple and interactive Gradio interface
* Fast local inference using TensorFlow

---

## Demo

Demo Video:
https://github.com/Allure815/image_quality_checker/blob/main/Demo-Image%20Classifier.mp4

Screenshot:
https://github.com/Allure815/image_quality_checker/blob/main/img-ss.png

---

## Tech Stack

* Python
* TensorFlow / Keras
* Gradio
* NumPy
* Pillow

---

## Project Structure

image_quality_checker
model/
  image_quality_model.h5
app.py
utils.py
requirements.txt
README.md

---

## How to Run the Project

1. Clone the repository

git clone [https://github.com/yourusername/image_quality_checker.git](https://github.com/yourusername/image_quality_checker.git)

cd image_quality_checker

2. Create a virtual environment

python -m venv venv

3. Activate the virtual environment

Windows

venv\Scripts\activate

4. Install dependencies

pip install -r requirements.txt

5. Run the application

python app.py

6. Open in browser

[http://127.0.0.1:7860](http://127.0.0.1:7860)

Upload an image to test the AI prediction.

---

## Model Behavior

The model outputs a prediction score between 0 and 1.

Score greater than 0.7 → Good Quality
Score between 0.4 and 0.7 → Medium Quality
Score less than 0.4 → Low Quality



## Future Improvements

* Add confidence score visualization
* Provide explanation for predictions
* Deploy the application online
* Improve model accuracy with more training data
