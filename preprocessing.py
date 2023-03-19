import numpy as np
import matplotlib.pyplot as plt

# audio processing/display
import soundfile as sf
import IPython
from scipy.io.wavfile import read
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_io as tfio



import numpy as np
import os

def prepare_data(name_dataset):
    """ load the data, convert it to a tensor
    
    Keyword arguments: 
    name_dataset - name of the folder the dataset was loaded into
    """
    data = []
    #sorted(files) TF suggestion
    for root, dirs, files in sorted(os.walk(name_dataset)):
        for filename in sorted(files):
            if filename.endswith('.wav'):
                # Load the audio file using librosa
                filepath = os.path.join(root, filename)
                audio, sr = librosa.load(filepath, sr=16000)
                # convert the audio input to a tensor of type float - float is important for the resampling
                audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
                # convert the sample rate, ito a integer tensor
                sr = tf.cast(sr, dtype=tf.int64)
                # Goes from sample_rate to 16000Hz - amplitude of the audio signal
                audio = tfio.audio.resample(audio_tensor, rate_in=sr, rate_out=16000)
                # save in list data
                data.append(np.array(audio))
    # Convert the data to a NumPy array
    return np.array(data)

def make_same(data, size):
    ''' make sure all data the same length
    
    Keyword arguments: 
    data = the dataset
    size - length all the data is normed to
    '''
    result = np.zeros((len(data), size), dtype=data[0].dtype)
    for i, d in enumerate(data):
        result[i, :min(size, d.shape[0])] = d[:min(size, d.shape[0])]
    return tf.stack(result)

def noise(data):
    ''' adding Gaussian noise to data
    
    Keyword arguments: 
    data = the dataset
    '''
    temp_data = []
    noise_factor = 0.004
    # add noise to the audio
    for i in data:
        temp = i + noise_factor*np.random.normal(size=i.shape)
        temp_data.append(temp)
    return temp_data

def spectrogram(data):
    ''' convert to a melspectrogram, normalize data, and add a faux-color dimension
    
    Keyword arguments: 
    data = the dataset
    '''    
    spec = []
    for i in range(len(data)):
        spectrogram = librosa.feature.melspectrogram(y=np.array(data[i]), sr = 16000, n_fft=1024, hop_length=256,n_mels = 128)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        # normalize pixel values to be between -1 and 1
        spectrogram = ((spectrogram+40) / 40) 
        # add a "color channel"
        spectrogram = np.expand_dims(spectrogram, axis=-1)
        spec.append(spectrogram)
    return np.array(spec)
