import json
import base64
import gradio as gr
from fastbook import *
import requests
from fastai.learner import load_learner
from pathlib import Path
from io import BytesIO
from PIL import Image

# Load model
model_url = "https://filedn.eu/lI9nnolEiqVYFksWq9Mm30u/export.pkl"

def download_model(url):
    response = requests.get(url)
    model_data = BytesIO(response.content)
    return model_data

def predict(image):
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

def lambda_handler(event, context):
    # Decode the image from base64
    image_data = base64.b64decode(event['body'])
    image = Image.open(BytesIO(image_data))

    # Call the predict function
    result = predict(image)

    # Return the result as a JSON response
    return {
        'statusCode': 200,
        'headers': { 'Content-Type': 'application/json' },
        'body': json.dumps({ 'result': result })
    }
