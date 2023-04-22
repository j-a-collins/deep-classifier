"""
This script creates a Gradio app to classify images of cats and dogs.

The app loads a simple neural net model ('export.pkl') and provides a user interface
for users to upload images. The app returns a classification result (cat, dog, or
"not dog, not cat" if the confidence is below 90%).

The Gradio app also includes custom text describing the model and a link to the
related blog: www.deepest-net.com.

Author: J-A-Collins
"""

import gradio as gr
from fastbook import *
import requests
from fastai.learner import load_learner
from pathlib import Path
from io import BytesIO

# Load model
# learn_inf = load_learner('export.pkl')
model_url = "https://filedn.eu/lI9nnolEiqVYFksWq9Mm30u/export.pkl"


def download_model(url):
    response = requests.get(url)
    model_data = BytesIO(response.content)
    return model_data


def predict(image):
    """
    Classify the input image as 'cat', 'dog', or, if the model
    confidence is below 90%, 'not dog, not cat'.

    Parameters:
        image: An image file to be classified by the model.

    Returns:
        A string containing the classification result.
    """
    res = learn_inf.predict(image)
    labels = ["cat", "dog"]
    probabilities = [float(probability) for probability in res[2]]
    max_probability = max(probabilities)

    if max_probability < 0.9:
        return "Not dog, not cat"

    label = labels[probabilities.index(max_probability)]
    probability_percentage = round(max_probability * 100, 2)
    return f"{label} ({probability_percentage}% confidence)"


model_data = download_model(model_url)
learn_inf = load_learner(model_data)

# Define Gradio input and output components
image_input = gr.inputs.Image(
    shape=(224, 224), label="Upload an image of a cat or a dog"
)
label_output = gr.outputs.Textbox(label="Prediction")

# Define custom text
description = """
This model was built using only 300 images total as the data set. You can read about it on my blog about deep learning and AI safety. It's called 'Deepest Net.' Here: <a href="https://www.deepest-net.com/" target="_blank">https://www.deepest-net.com/</a>
"""

# Define the Gradio interface with a title, custom text, and input/output components
gr.Interface(
    fn=predict,
    inputs=image_input,
    outputs=label_output,
    title="Deep's Cat/Dog Classifier",
    description=description,
    capture_session=True,
).launch()
