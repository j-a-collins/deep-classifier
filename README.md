# deep-classifier
A repo for deploying my teeny tiny neural net trained to classify images as either dogs or cats, or neither.

![CatDog GIF](classifier.gif)

The model is built using Fastai and then deployed as a Gradio app. To use the classifier, follow the steps below:

## Getting Started

1. Make sure you have Python 3.6 or higher installed on your machine. You can download the latest version of Python from the [official website](https://www.python.org/downloads/).

2. Install the required packages:
'''pip install fastai gradio'''

3. Clone this repository: git clone https://github.com/yourusername/cat-dog-classifier.git
cd cat-dog-classifier


## Build the Model

1. Open the `classifier.ipynb` notebook using Jupyter Notebook, JupyterLab, or your preferred notebook platform.

2. Run the cells in the notebook to build the image classifier model. This will create the 'export.pkl' file containing the trained model.

## Deploy the Gradio App

1. Run the `deploy.py` script to launch the Gradio app: python deploy.py

This will start a local server and provide you with a URL to access the app in your web browser.

2. Open the URL in your web browser to use the Cat-Dog Image Classifier. Upload an image of a cat or a dog, and the classifier will display the prediction along with the confidence level.


