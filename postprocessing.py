# -*- coding: utf-8 -*-
"""postprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1brDRob4BCWGUucbrGF6bFZg63uAiuK0z
"""

import tensorflow as tf
import numpy as np
import librosa

def convert_to_audio(data, noisy):
  ''' reconvert prediction to audio
  
    Keyword arguments:
    dataset -- the dataset to be sampled from
  '''
  #reverse the normalisation
  data = data*40-40 +20
  #get the phase information
  spectrogram = librosa.stft(noisy, n_fft=2048, hop_length=256, window='hann')
  phase = np.angle(spectrogram)
  #convert the mel-spectrogram to a stft
  mag = librosa.feature.inverse.mel_to_stft(data, sr=16000)
  #convert the magnitude spectrogram into a complex spectorgram
  spectrogram_complex = mag * np.exp(1j * np.angle(phase))
  #convert the complex spectrogram to audio
  reconstructed_audio=librosa.istft(spectrogram_complex ,hop_length=256)
  #returns the audio data
  return reconstructed_audio
  
