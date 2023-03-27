import os
from scipy.io.wavfile import read
import subprocess
package_name = 'tensorflow-io'
subprocess.check_call(["pip", "install", package_name])

import matplotlib.pyplot as plt
import numpy as np
import IPython
import librosa
import librosa.display
import soundfile as sf
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_io as tfio


def prepare_data(name_dataset, size):
    """Loads audio files from the given dataset folder, converts them to tensors,
       and pads them to the same length.
    
    Args:
        name_dataset (str): Name of the folder that contains the dataset.
        size (int): The target length of the audio samples after normalization.
        
    Returns:
        A tensor of shape (len(data), size) representing the audio data, where each row corresponds to an audio sample.
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

    result = np.zeros((len(data), size), dtype=data[0].dtype)
    for i, d in enumerate(data):
        result[i, :min(size, d.shape[0])] = d[:min(size, d.shape[0])]
    return tf.stack(result)
        

def noise(data, noise_factor=0.004):
    ''' Adds random noise to the audio data.

    Args:
        data (list of numpy arrays): A list of audio samples, where each sample is represented as a numpy array.
        noise_factor (float, optional): The factor controlling the amount of noise to add to the audio data. Default is 0.004.

    Returns:
        A list of numpy arrays, where each array represents the audio data with added noise.
    
    '''
    temp_data = []
    noise_factor = noise_factor
    # Add noise to the audio
    for i in data:
        temp = i + noise_factor*np.random.normal(size=i.shape)
        temp_data.append(temp)
    return temp_data

def spectrogram(data, sr=16000, n_fft=1024, hop_length=256, n_mels=128):
    ''' Computes the Mel spectrogram of the audio data.
    
    Args:
        data (list of numpy arrays): A list of audio samples, where each sample is represented as a numpy array.
        sr (int, optional): The sample rate of the audio data. Default is 16000.
        n_fft (int, optional): The number of FFT points. Default is 1024.
        hop_length (int, optional): The number of samples between successive frames. Default is 256.
        n_mels (int, optional): The number of Mel bands to generate. Default is 128.
        
    Returns:
        A tensor of shape (len(data), n_mels, T, 1), where each element is a float representing the Mel spectrogram
        of the corresponding audio sample. T is the number of time frames in the spectrogram, which is determined
        by the length of the input audio.
    
    '''    
    spec = []
    for i in range(len(data)):
        spectrogram = librosa.feature.melspectrogram(y=np.array(data[i]), sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        # Normalize pixel values to be between -1 and 1
        spectrogram = ((spectrogram+40) / 40) 
        # Add a "color channel"
        spectrogram = np.expand_dims(spectrogram, axis=-1)
        spec.append(spectrogram)
    return np.array(spec)
