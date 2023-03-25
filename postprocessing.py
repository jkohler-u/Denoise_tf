# -*- coding: utf-8 -*-
"""postprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1brDRob4BCWGUucbrGF6bFZg63uAiuK0z
"""

import tensorflow as tf

def get_one_of_each(dataset):
  '''produces one sample to compare the quality of the prediction 
  
   Keyword arguments:
   dataset -- the dataset to be sampled from
   '''
  noisy, clean = [0], [0]
  #samples the first datapoint fom the dataset
  for x, y in dataset.take(1).as_numpy_iterator():
      noisy = x[:1]
      clean = y[:1] 
  #procduces a dataset with one dataset for the prediciton
  prediction_for_rest = tf.data.Dataset.from_tensor_slices((noisy, clean))
  #retruns the dataset, as well as its componets in a displayable format
  return prediction_for_rest, tf.squeeze(noisy), tf.squeeze(clean)

import librosa

def convert_to_audio(data):
  ''' reconvert prediction to audio
  
    Keyword arguments:
    dataset -- the dataset to be sampled from
  '''
  #reverse the nromalisation
  S = data*40-40 +20
  #convert mel-spectogram to audio
  S = librosa.db_to_power(S, ref=1.0)
  S = librosa.feature.inverse.mel_to_audio(S, sr=16000, n_fft=1024,hop_length=256, win_length= 1024)
  #returns the audio data
  return S