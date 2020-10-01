import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import base64

import tensorflow
from librosa.display import specshow
from librosa.core import power_to_db
from librosa.feature import melspectrogram


CLASSES = [
    'air_conditioner', 'car_horn', 'children_playing','dog_bark', 'drilling', 
    'engine_idling', 'gun_shot', 'jackhammer', 'siren','street_music'
]

def envelope(x, rate, threshold):
    y = pd.Series(x).apply(np.abs)
    y_mean = y.rolling(window=int(rate/5), min_periods=1, center=True).mean()
    mask = y_mean > threshold
    return x[mask]

def encode_mpl_fig(fig, ax):
    buf = io.BytesIO()
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(fname=buf, format="png", transparent=True, bbox_inches=extent, pad_inches=0)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return "data:image/png;base64,{}".format(data)

def spectrogram_image(mels):
    fig = Figure()
    ax = fig.add_subplot(1,1,1)
    specshow(data=power_to_db(mels), ax=ax, sr=22050, cmap='Blues')
    return encode_mpl_fig(fig, ax)

def shape_mels(vec):
    max_pad_len = 174
    if vec.shape[1] > max_pad_len:
        center = vec.shape[1] // 2
        vec = vec[:, (center - max_pad_len//2):(center + max_pad_len//2)]
    pad_width = max_pad_len - vec.shape[1]
    vec = np.pad(vec, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return vec.reshape(1, vec.shape[0], vec.shape[1], 1)

def format_output(preds):
    pred_proba = "{:.1f}".format(np.amax(preds)*100) 
    pred_class = str(CLASSES[np.argmax(preds)]).replace('_', ' ').title()
    return {'result': pred_class, 'probability': pred_proba}

def featurize_predict(x, model):
    mels_raw = melspectrogram(y=envelope(x, 22050, 0.00005), n_mels=60)
    mels_clean = shape_mels(mels_raw)
    output = format_output(model.predict(mels_clean))
    output['spec'] = spectrogram_image(mels_raw)
    return output
