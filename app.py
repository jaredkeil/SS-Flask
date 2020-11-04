from src.helpers import featurize_predict

from flask import Flask, url_for, request, render_template, Response, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import librosa
from tensorflow.python.keras.models import load_model

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


app = Flask(__name__)

MODEL = load_model(filepath='models/model1.hdf5')
MODEL.summary()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get user's audio from post request
        print("audio finished")
        if 'file' not in request.files:
            app.logger.debug("[server] no file part")
            return '[server] no file part'
        r = request.files['file']
        try:
            signal, _ = librosa.load(r)
        except ValueError:
            return jsonify(result="inputError")

        mp = featurize_predict(signal, MODEL)
        return jsonify(result=mp['result'], probability=mp['probability'], spec=mp['spec'])

    return None

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')