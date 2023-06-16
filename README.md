# Coding3FinalProject
Git repository for Coding 3 - final project


For my final project, I decided to explore how autoencoders can be used to recreate and generate content in video games. More specifically, I used a dataset of non-player character images from Nintendo's popular game *Animal Crossing: New Horizons*. My aim was to build an autoencoder that could recreate the images and construct a latent space that would allow me to generate new images. 

I found a dataset on Kaggle that contains images of all of the villagers from the game, with 392 files total. The dataset can be found at the following link: https://www.kaggle.com/datasets/jahysama/animal-crossing-new-horizons-all-villagers

## Sources 

This project uses code from the following sources:
1. *Building Autoencoders in Keras*, https://blog.keras.io/building-autoencoders-in-keras.html
2. The 'ConvolutionalVAE' notebook from week 4 of class
3. ChatGPT (especially for debugging, image processing and visualisation)
4. Help from the coding 3 team at the CCI 

## Data preprocessing

Because the original images were of all different sizes, I needed to resize the images to begin with. I chose the dimensions (640, 400) and with help from ChatGPT, was able to reshape all of these images to these dimensions. I then was able to create the training and test set from this dataset, with around 20% of the images belonging to the test set. Since this is an unsupervised learning problem, there are no arrays for labels.

One issue I noticed right at the start is that the photos are not all aligned. For instance, some animals (such as a mouse or cat) are staring straight at the 'camera', whereas others (such as elephants) are looking off to the side. This isn't ideal for an autoencoder, and is something important to keep in mind. 

## Autoencoders

To build the autoencoders, I followed along the Keras tutorial (https://blog.keras.io/building-autoencoders-in-keras.html). For the variational autoencoder, I also drew from the ConvolutionalVAE notebook from week 4 of class.

In order to debug, visualise the results, explore the latent space, generate random latent vectors, etc. I also used ChatGPT to help with the code. 

### Simple autoencoder

I decided to start small with the simplest autoencoder possible from the tutorial, both to get a sense of the potential of an autoencoder with this specific dataset, but also to build up on my own knowledge. This autoencoder just creates a simple encoder and decoder model with a dense layer each, and is compiled with relatively standard settings (an Adam optimiser and a binary crossentropy loss function).

With help from ChatGPT, I was able to write code that visualises the results. I ran the autoencoder on the test results, and chose 5 to plot. The row above are the original images, and the entry below is its corresponding autoencoder reconstruction.

Unsurprisingly, the results are not very accurate and overall are quite blurry. However, I find that they still hold valuable information. I could make out the various forms of repetitive characters, such as the cat figures, which emphasises their prevalence in the training set. 

![simplest autoencoder 1 results](https://git.arts.ac.uk/storage/user/650/files/538956c7-4a41-46cb-acf4-da747eb9c5da)

Given this is a very simple autoencoder, I was excited to see how more sophisticated models would reconstruct the data.

### Convolutional autoencoder - Part 1

I continued with the tutorial to build a convolutional autoencoder. This model uses 

### Convolutional autoencoder - Part 2

### Variational autoencoder

## Results 

## Main challenges 

## Future avenues 
