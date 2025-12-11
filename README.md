
## AUTOENCODERS-FOR-IMAGE-SUPER-RESOLUTION

A Convolutional Autoencoder Implementation in TensorFlow/Keras

This repository contains the complete implementation and documentation for an image super-resolution autoencoder, including:

A detailed 12-page tutorial (AutoEncoder.pdf) explaining theory + practical implementation.

Python code (autoencoder.py) implementing a convolutional autoencoder using low-resolution ↔ high-resolution paired images.

Dataset loading pipeline using a Keras Sequence class.

Training logs, architecture summary, loss curves, and reconstruction outputs.
 
## **Repository Structure**
1.autoencoder.py            
2.AutoEncoder.pdf           
3.image_data.csv            
4.low res/                  
5.high res/                 
6.README.md                 
7.LICENSE                 


## **Project Overvie**

This project builds a Convolutional Autoencoder designed to convert low-resolution images into high-resolution reconstructions.

# **The architecture includes:**

# **Encoder**

Conv2D(16) → MaxPooling

Conv2D(32) → MaxPooling

Conv2D(64) → MaxPooling

# **Decoder**

Conv2D(64) → UpSampling

Conv2D(32) → UpSampling

Conv2D(16) → UpSampling

Conv2D(3, sigmoid)

Total trainable parameters: 84,035.

## **Installation**
**1. Clone the Repository**
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>




# **Recommended packages:**
tensorflow
numpy
pandas
matplotlib
scikit-learn

## **Dataset Format**

**Your dataset must have:**

CSV File (image_data.csv)

**Columns:**

low_res, high_res


**Example:**

1_2.jpg,1.jpg
2_2.jpg,2.jpg

**Folders**
BASE_FOLDER/
 ├── low res/
 └── high res/


The script automatically attaches full paths to each filename.

## **Running the Model**
Training

**Inside autoencoder.py:**

history = autoencoder.fit(
    train_seq,
    steps_per_epoch=50,
    validation_data=val_seq,
    validation_steps=10,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)


This trains the model using:

Loss: MSE

Optimizer: Adam

Batch size: 8

EarlyStopping + ModelCheckpoint + ReduceLROnPlateau

## **Visual Outputs**

The script produces:

**1. Training vs Validation Loss Plot**

Shows steady convergence 

**2. Reconstruction Grid**

Low-res → High-res → Predicted


**To generate:**

autoencoder.predict(low_batch)
## **Evaluation Summary**

Based on results described in your PDF:

Strengths

Good detail and colour preservation

Stable training

Lightweight model (84k params)

Limitations

MSE leads to soft or slightly blurry textures

No skip connections (thus some detail is lost)

GAN-based models would perform better on fine textures

## **Tutorial (PDF)**

A full written tutorial is included:
AutoEncoder.pdf 

AutoEncoder

**It contains:**

Intro to autoencoders

Types (denoising, sparse, VAE, convolutional)

Architecture diagrams

Dataset explanation

Model summary (page 8)

Training/validation plots

Reconstruction outputs

Conclusion + references

## **Future Improvements**

Add skip connections (U-Net style)

Replace MSE with perceptual loss (VGG-based)

Use GAN architecture (SRGAN or ESRGAN)

Try residual autoencoders
## **HOW TO RUN THIS FILE**

Using your uploaded file: autoencoder (1).py 

autoencoder (1)

1️⃣ **Open Terminal / CMD inside the folder**

Example:

cd "C:\Users\YourName\Downloads\AutoencoderProject"

2️⃣ **Install required libraries**

Run:

pip install tensorflow matplotlib pandas numpy scikit-learn pillow

3️⃣ **Make sure your dataset is in correct Google Drive path**

Your code expects:

/content/drive/MyDrive/Image Super Resolution - Unsplash/
    ├── image_data.csv
    ├── low res/
    └── high res/


So in Colab, use:

from google.colab import drive
drive.mount('/content/drive')

4️⃣ **Run the script**
✔ If running in Google Colab

Upload the file → open it → run all cells.

✔ If running locally

Remove the Google Drive lines and run:

python "autoencoder (1).py"


## **License**

This project is licensed under the MIT License - see the LICENSE file for details.

