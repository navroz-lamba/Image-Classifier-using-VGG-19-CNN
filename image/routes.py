from image import app
from flask import render_template, request, redirect
import numpy as np
import sys, os, glob, re
from image import model_predict, model
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from the post
        f = request.files['file']
        # save the files to uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename)
        )
        f.save(file_path)
        # Here we make prediction
        pred = model_predict(file_path, model)
        # now we will have to decode 
        pred_class = decode_predictions(pred, top=1)
        result = str(pred_class[0][0][1])

        return result
    else:
        return render_template("index.html")