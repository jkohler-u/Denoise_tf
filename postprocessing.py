# -*- coding: utf-8 -*-
"""postprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1brDRob4BCWGUucbrGF6bFZg63uAiuK0z
"""

import tensorflow as tf
import numpy as np
import librosa

def convert_to_audio(data, noisy, n_fft=2048, hop_length=256, window='hann', sr=16000):
    ''' reconvert prediction to audio
  
      Keyword arguments:
          data: the predicted magnitude spectrogram to be sampled from
          noisy: the original noisy audio
      Arguments:
           n_fft: length of the windowed signal after padding with zeros (default is 2048)
           hop_length: number of audio samples between adjacent STFT columns (default is: 256)
           window: window function (default is  a raised cosine window: 'hann')
       Returns: 
           Reconstructed audio as np.ndarray
           
    '''
    #we don't reverse the normalization here, because it sounds better that way
    #get the phase information
    spectrogram = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length, window=window)
    phase = np.angle(spectrogram)
    #convert the mel-spectrogram to a stft
    mag = librosa.feature.inverse.mel_to_stft(data, sr=sr)
    #convert the magnitude spectrogram into a complex spectorgram
    spectrogram_complex = mag * np.exp(1j * np.angle(phase))
    #convert the complex spectrogram to audio
    reconstructed_audio=librosa.istft(spectrogram_complex ,hop_length=hop_length)
    #returns the audio data
    return reconstructed_audio
  
