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

from tensorflow.keras.models import load_model
import librosa
from librosa.display import specshow

# Declare a flask app
application = Flask(__name__)


# Model saved with Keras model.save()
MODEL_PATH = 'models/model1.hdf5'

# Load your own trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

pred2class = {0: 'air_conditioner',
            1: 'car_horn',
            2: 'children_playing',
            3: 'dog_bark',
            4: 'drilling',
            5: 'engine_idling',
            6: 'gun_shot',
            7: 'jackhammer',
            8: 'siren',
            9: 'street_music'}

def envelope(y, rate, threshold):

    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/5), min_periods=1, center=True).mean()
    mask = y_mean > threshold
    return mask


@application.route('/plot', methods=['GET', 'POST'])
def plot_spectrogram(mels):
    if request.method == 'POST':
        print('posting to spect')
    
    # Generate plot
    fig = Figure()
    axis = fig.subplots()
    mels = librosa.core.power_to_db(mels)
    specshow(mels, ax=axis, sr=22050, cmap='magma')
    buf = io.BytesIO()
 
    fig.savefig(buf, format="png", transparent=True)
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")

    return "data:image/png;base64,{}".format(data)


def model_predict(x, model):

    # masking of x
    x = x[envelope(x, 22050, 0.00005)]
    vec = librosa.feature.melspectrogram(x, n_mels=60)
     #send to plot_spectrogram before reshaping
    plot = plot_spectrogram(vec)
    #shape for model, trim /pad as necessary
    max_pad_len = 174
    if vec.shape[1] > max_pad_len:
        center = vec.shape[1] // 2
        trim = vec[:, (center - max_pad_len//2):(center + max_pad_len//2)]
    pad_width = max_pad_len - vec.shape[1]
    vec = np.pad(vec, pad_width=((0, 0), (0, pad_width)), mode='constant')
    vec = vec.reshape(1, vec.shape[0], vec.shape[1], 1)

    preds = model.predict(vec)
    # print("preds shape:", preds.shape)
    return preds, plot


@application.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@application.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the audio from post request
        print("audio finished")
        if 'file' not in request.files:
            application.logger.debug("[server] no file part")
            return '[server] no file part'
        
        r = request.files['file']
        signal, _ = librosa.load(r)

        # Make prediction
        preds, plot = model_predict(signal, model)

        # Process your result for human
        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        pred_class = pred2class[np.argmax(preds)] 

        result = str(pred_class)               # Convert to string
        result = result.replace('_', ' ').capitalize()
        
        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba, plot=plot)

    return None


if __name__ == '__main__':
    application.debug = True
    application.run()
    