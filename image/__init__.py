import flask
from flask import Flask, url_for
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# instantiating a class object 
app = Flask(__name__)

model_path = 'vgg19.h5'

# load the model 
model = load_model(model_path)
# model._make_predict_function()


# preprocessing function
def model_predict(img_path, model):
    # load the image and set the size to 224,224
    img = image.load_img(img_path, target_size=(224,224))
    # change the image to array 
    x = image.img_to_array(img)
    # add dimension so we could pass it as an input to the network
    x = np.expand_dims(x, axis=0)
    # scale the input
    x = preprocess_input(x)
    # make predictions
    preds = model.predict(x)
    
    return preds

from image import routes