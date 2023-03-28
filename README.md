# Denoise_tf

This is a 2D-spectrogram-U-Net trained to denoise audio books.

For this project, we worked with the Microsoft Scalable Noisy Speech Dataset. These are the links where you can download the necessary files:

Clean Data: https://drive.google.com/file/d/1KjGWrN8VzUwAmxA9g0hy2423mrqfM6Uq/view?usp=share_link

Noisy Data: https://drive.google.com/file/d/1tloBlhRN4Aa7TFu1e1eQ_NpRipg0Ggkq/view?usp=share_link

You can upload these files in your Google Drive and then run the Model.ibynp. You will be asked for your permission to connect to your Drive and if this permission is given the audio files can be loaded and you can let the model run.

In the beginning of our project we worked with the Librivox Dataset. It is not necessary to run this code but, if you like, you can download the clean audios and add noise with the Gaussian noise function from the preproccessing to it: https://drive.google.com/file/d/1wrQgdD3-4oHmqpQRzqbCEU7v9nYYu3mJ/view?usp=share_link

To validate how our model works with unknown data we created this dataset of noisy audios: https://drive.google.com/file/d/185gnGWZhxIkQVkx5pUmZ-roebFEP9rjQ/view?usp=share_link. We tested unknown speakers, languages and noise. You can download this dataset and after running the model try out how the model can deal with unknown data.




This project was created as a final project for the course "Implementing ANNs with TensorFlow" of the Cognitive Science program of the Universität Osnabrück.
Additional information to our research, methods, results and a discussion can be found in the paper. The video serves to explain how our model works illustratively.
The video can be found here: https://drive.google.com/file/d/1A3N_yXVaOfmSAh7RvcpCjlRrfeX0pHzL/view?usp=share_link
