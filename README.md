# Music Genre Classifier

This repository contains a Music Genre Classifier built using the GTZAN dataset. The dataset can be downloaded from <a href="https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download" target="_blank">GTZAN dataset-Kaggle</a>.

## Overview
Music Genre Classification is a fundamental task in the field of music information retrieval. This project aims to classify audio tracks into one of ten genres: Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, and Rock. The GTZAN dataset, with 1000 audio tracks each 30 seconds long, sampled at 22050Hz, serves as the foundation for this classifier.

## Gradio Integration
[Gradio](https://www.gradio.app/) is used in this project for building a simple and easy-to-use interface to interact with the Music Genre Classifier. Gradio allows users to upload an audio file and get the predicted genre along with the probability distribution.

## Dataset
The GTZAN dataset is used for this classification task. It consists of 1000 audio tracks, each 30 seconds long, sampled at 22050Hz, and each belonging to one of the following genres:

- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

## Usage
1. **Download and Extract the Dataset:**
   - Download the GTZAN dataset from [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download) (opens in new tab).
   - Extract the dataset into the `data` folder.

2. **Training the Classifier:**
   - If using Google Colab, mount your drive:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Run the `music_genre_classifier.ipynb` Jupyter Notebook to train and test the classifier.
   - `data_10.json` file will be generated which has MFCC values, mapping, and labels of genres.

3. **Saving the Model:**
   - To save the trained model in `.h5` format, use the following steps in your notebook:
     ```python
     import os
     import tensorflow as tf
     from tensorflow import keras
     import tensorflow.keras.models as models

     # Save the model in HDF5 format
     models.save_model(model, '/content/drive/MyDrive/saved_models/music_cnn.h5')
     ```

4. **Testing with New Data:**
   - To test with a new audio file:
     - Install required packages:
       ```bash
       pip install gradio
       pip install keras
       ```
     - Upload `mapping.json` file to Colab.
     - Run `new_data.ipynb`. It will generate a link to the Gradio interface.
     - In the Gradio interface, upload a `.wav` file of 30 seconds duration to get its predicted genre and distribution.

## Requirements
- Python 3.x
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Librosa
- Gradio
- TensorFlow

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



     
 
