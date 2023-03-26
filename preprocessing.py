import subprocess
# Define the package you want to install
package_name = 'tensorflow-io'
# Use pip to install the package
subprocess.check_call(["pip", "install", package_name])

import tensorflow as tf
import tensorflow_io as tfio

import os
import numpy as np
import matplotlib.pyplot as plt

# Audio processing/display
import IPython
import librosa
import librosa.display
from scipy.io.wavfile import read
import soundfile as sf
from sklearn.model_selection import train_test_split

        
def prepare_data(name_dataset):
    """ load the data, convert it to a tensor
    
    Keyword arguments: 
    name_dataset - name of the folder the dataset was loaded into
    """
    data = []
   
    for root, dirs, files in sorted(os.walk(name_dataset)):
        for filename in sorted(files):
            if filename.endswith('.wav'):
                # Load the audio file using librosa
                filepath = os.path.join(root, filename)
                audio, sr = librosa.load(filepath, sr=16000)
                # Convert the audio input to a tensor of type float - float is important for the resampling
                audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)              
                # Save in list data
                data.append(np.array(audio_tensor))
    return data

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
    # Add noise to the audio
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
        # Normalize pixel values to be between -1 and 1
        spectrogram = ((spectrogram+40) / 40) 
        # Add a "color channel"
        spectrogram = np.expand_dims(spectrogram, axis=-1)
        spec.append(spectrogram)
    return np.array(spec)
