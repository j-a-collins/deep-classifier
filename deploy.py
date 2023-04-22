"""
This script creates a Gradio app to classify images of cats and dogs.

The app loads a simple neural net model ('export.pkl') and provides a user interface
for users to upload images. The app returns a classification result (cat, dog, or
"not dog, not cat" if the confidence is below 90%).

The Gradio app also includes custom text describing the model and a link to the
related blog: www.deepest-net.com.

Author: J-A-Collins
"""

import json
import base64
import gradio as gr
from io import BytesIO
import requests

API_URL = "https://yfskn8ycx8.execute-api.eu-north-1.amazonaws.com/prod/gradioapp"

def predict(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    payload = json.dumps({"image": img_base64})
    response = requests.post(API_URL, data=payload, headers={"Content-Type": "application/json"})

    if response.status_code == 200:
        result = response.json().get("result")
        return result
    else:
        return f"Error: {response.status_code} - {response.text}"

image_input = gr.inputs.Image(shape=(224, 224), label="Upload an image of a cat or a dog")
label_output = gr.outputs.Textbox(label="Prediction")

description = """
This model was built using only 300 images total as the data set. You can read about it on my blog about deep learning and AI safety. It's called 'Deepest Net.' Here: <a href="https://www.deepest-net.com/" target="_blank">https://www.deepest-net.com/</a>
"""

gr.Interface(
    fn=predict,
    inputs=image_input,
    outputs=label_output,
    title="Deep's Cat/Dog Classifier",
    description=description,
    capture_session=True,
).launch()
