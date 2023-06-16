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

I continued with the tutorial to build a convolutional autoencoder. This model uses Conv2D and MaxPooling2D layers at the encoder level, and Conv2D and UpSampling layers at the decoder level. After compiling and training, the results were promising. The final loss value in training was 0.3554 and the val_loss was 0.3542. The results were also noticeably more accurate, despite a hiccup with the colour channels:

![convolutional autoencoder 2](https://git.arts.ac.uk/storage/user/650/files/4b8c7453-8468-4d37-a074-15489af0fac7)

The overall forms are a lot more defined, and it is easy to distinguish between the different animals. However, after discussing with the coding 3 team, I realised that the latent space is actually quite large and so is not compressed enough to generalise to new images. The input images were of the shape (None, 640, 400, 4) and only get compressed to (None, 80, 50, 8).  

<img src="https://git.arts.ac.uk/storage/user/650/files/9d3e8334-f5be-4ca4-aa0b-a5d61aaac6ed" width="45%">

To fully explore the potential of autoencoders, I decided to build on this latest model by compressing the data down even further. 

### Convolutional autoencoder - Part 2

With help from the coding 3 team and ChatGPT, I was able to build a new model that flattens the data down to a more compressed format. This was acheived by using the Flatten() command and adding Dense() layers to the model, as well as using Reshape() at the decoder level.

<img src="https://git.arts.ac.uk/storage/user/650/files/4eb414c6-9425-41ec-a804-e8decf060198" width="45%">


Despite the compressed latent vector, the reconstructed images still keep the outline of the animal and are more defined than the simplest autoencoder. They are definitely 'grainier' in appearance than the first convolutional autoencoder, and it is more difficult to make out the features of the animal. 

![convolutional autoencoder 2 results](https://git.arts.ac.uk/storage/user/650/files/3a5bdec2-418e-4da7-8bbb-9470c74b6ca3)

That being said, I was pleasantly surprised with the performance of this autoencoder. Given the compressed latent vector, the model was still able to reconstruct the original image to a rather large extent. 

### Variational autoencoder

Finally, I built a variational autoencoder to use on this dataset. Again, I followed the tutorial and also drew from the week 4 notebook 'ConvolutionalVAE'. 

Initilly, the dataset didn't work with this model due to its shape. With help from ChatGPT, I was able to get the data flattened down to a (None, 1024000) representation that worked with the autoencoder. 

<img src="https://git.arts.ac.uk/storage/user/650/files/84b3fa22-2c05-4cd5-b4eb-b5e948101af2" width="45%">

After training and compiling, I immediately noticed that the loss was always 'NaN' (not a number). This was confirmed when I tried to visualise the reconstructed images, and nothing appeared. There was no error message in the code, so it wasn't obvious what had gone wrong.

ChatGPT gave me advice on what to check. Two of these were setting the correct learning rate and decreasing the batch size. I created a separate Adam optimiser object that set the learning rate to 0.0001 and decreased the batch size to 16. These changes worked, and when training the model a second time, I saw a proper loss value (albeit a very large one).

![vae model results](https://git.arts.ac.uk/storage/user/650/files/5929ce82-0115-490b-bc47-e5b898fc6dac)

The results here are more similar to the simple autoencoder. The outlines are generally quite blurry, and seem to be a blend together different animal forms (for example, the fourth one appears to have cat and mouse features). Additionally, there seems to be some interesting behaviour with the eagle. In the first and third example, the outline of an eagle is easy to find, and is a little blurrier in the other examples. 

### Generating new content

Using the variational autoencoder, I explored how to generate new images using the latent space. First, I chose a random test image and ran it through the encoder  to get its latent vector, then passed this into the decoder to get the image reconstruction.

<img src="https://git.arts.ac.uk/storage/user/650/files/79df1659-bf4c-4149-8fa8-ca1fe2d1deac" width="20%"> <img src="https://git.arts.ac.uk/storage/user/650/files/253fbc3f-db91-433d-b7f1-b3edf8c25c95" width="20%">

To generate new latent vectors, I first wanted to get a sense of where the training latent vectors lay. With help from ChatGPT, I found the min and max latent vector values from the training data. I then generated 10 random vectors within this range, and ran them through the decoder.

<img src="https://git.arts.ac.uk/storage/user/650/files/db3c6d5e-1cfa-40ea-95e8-b140c185ea82" width="20%">

![vae random results](https://git.arts.ac.uk/storage/user/650/files/7d46b6b0-91e7-4f94-aff9-1760ae7a1bee)


## Evaluating performance  

conv2d one overall was great

VAE/Simple autoencoder similar, makes me think there needed to be more convolutional layers to truly capture the details of the dataset

## Main challenges 

## Future avenues 
