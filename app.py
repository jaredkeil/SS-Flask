import os
import sys

# Flask
from flask import Flask, url_for, request, render_template, Response, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import base64

import tensorflow as tf
from tensorflow.python.keras.models import load_model
import librosa
from librosa.display import specshow

# Declare a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model1.hdf5'

# Load your own trained model
model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

pred2class = ['air_conditioner', 'car_horn', 'children_playing',
            'dog_bark', 'drilling', 'engine_idling', 'gun_shot',
            'jackhammer', 'siren','street_music']

def envelope(y, rate, threshold):
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/5), min_periods=1, center=True).mean()
    mask = y_mean > threshold
    return mask


# @app.route('/plot', methods=['GET', 'POST'])
def plot_spectrogram(mels):
    if request.method == 'POST':
        print('posting to spect')
    # Generate plot
    fig = Figure()
    axis = fig.subplots()
    mels = librosa.core.power_to_db(mels) # better visualization
    # put spectrogram into plt fig memory
    specshow(mels, ax=axis, sr=22050, cmap='magma')
    aspect_ratio = 1000
    axis.set_aspect(.5)
    buf = io.BytesIO()
    # write spectrogram image into buffer
    fig.savefig(buf, format="png", transparent=True, dpi=100, )
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")

    return "data:image/png;base64,{}".format(data)


def model_predict(x, model):
    # masking of x (removing quiet parts)
    x = x[envelope(y=x, rate=22050, threshold=0.00005)]
    vec = librosa.feature.melspectrogram(x, n_mels=60)
    # create spectrogram of audio before reshaping for prediction
    spec = plot_spectrogram(vec)
    # shape for model, trim/pad as necessary
    max_pad_len = 174
    if vec.shape[1] > max_pad_len:
        center = vec.shape[1] // 2
        vec = vec[:, (center - max_pad_len//2):(center + max_pad_len//2)]
    pad_width = max_pad_len - vec.shape[1]
    vec = np.pad(vec, pad_width=((0, 0), (0, pad_width)), mode='constant')
    vec = vec.reshape(1, vec.shape[0], vec.shape[1], 1)

    preds = model.predict(vec)
    # print("preds shape:", preds.shape)
    return preds, spec


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global app
    if request.method == 'POST':
        # Get the audio from post request
        print("audio finished")
        if 'file' not in request.files:
            app.logger.debug("[server] no file part")
            return '[server] no file part'
        
        r = request.files['file']
        signal, _ = librosa.load(r)

        # featurize input and get model's prediction
        preds, spec = model_predict(signal, model)

        # Process result for easy visibility
        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        pred_class = pred2class[np.argmax(preds)] 

        result = str(pred_class)
        result = result.replace('_', ' ').capitalize()
        
        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba, spec=spec)

    return None


if __name__ == '__main__':
    app.debug = True
    app.run()