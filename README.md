# Denoise_tf

This is a 2D-spectrogram-U-Net trained for denoising audio books.

For this project, we worked with the Microsoft Scalable Noisy Speech Dataset. These are the links where you can download the necessary files:

Clean Data: https://drive.google.com/file/d/1KjGWrN8VzUwAmxA9g0hy2423mrqfM6Uq/view?usp=share_link

Noisy Data: https://drive.google.com/file/d/1tloBlhRN4Aa7TFu1e1eQ_NpRipg0Ggkq/view?usp=share_link

You can upload these files in your Google Drive and then run the Model.ibynp. You will be asked for your permission to connect to your Drive and if this permission is given the audio files can be loaded and you can let the model run!

If you want to see how the trained model performs on data from a different dataset, you'll also need to download this one: https://drive.google.com/file/d/185gnGWZhxIkQVkx5pUmZ-roebFEP9rjQ/view?usp=share_link

This last file is the one from the Librivox Dataset we used in the beginning when we only added Gaussian noise to it. It is not necessary to run this code. https://drive.google.com/file/d/1wrQgdD3-4oHmqpQRzqbCEU7v9nYYu3mJ/view?usp=share_link

If you want to try these alternative datasets you have to adjust the filenames in the model.ibynp-code for opening and extracting the zipfiles.

