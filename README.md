🍎 Fruit and Vegetable Waste Detection using Machine Learning

📌 Project Overview
Food waste is a major global issue, especially in the supply chain of fruits and vegetables where spoilage occurs quickly. This project uses Machine Learning and Computer Vision to automatically detect whether fruits and vegetables are fresh or spoiled, helping reduce waste in markets, warehouses, and supply systems.

The model analyzes images of, bread,cheese,fruits and vegetables and classifies them into fresh or rotten categories.

🎯 Objectives

Detect spoiled fruits and vegetables automatically

Reduce food waste through early identification

Assist farmers, retailers, and supply chain managers

Build an AI-based classification system using image data

🧠 Technologies Used

Python

Machine Learning

Deep Learning

Computer Vision

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

Scikit-learn

📂 Dataset

The dataset consists of images of different fruits and vegetables categorized as:

Fresh

Rotten

-Example categories include:

Apples

Bananas

Tomatoes

Potatoes

Oranges

Images were preprocessed and resized before training the model.

⚙️ Project Workflow

Data Collection

Collect image image datasets.

Data Preprocessing

Image resizing

Normalization

Label encoding

Model Training

Convolutional Neural Network (CNN) used for image classification.

Model Evaluation

Accuracy

Confusion Matrix

Validation Loss

Prediction

The trained model predicts whether the item is Fresh or Rotten.

🏗 Model Architecture

The model is based on a Convolutional Neural Network (CNN) consisting of:

Convolution Layers

ReLU Activation

MaxPooling Layers

Fully Connected Layers

Softmax Output Layer

This architecture helps extract visual features such as texture, color, and surface damage.

📊 Results

Model Accuracy: 84%

Successfully detects spoiled produce from image input.

Can be extended for real-time detection using a camera.

🚀 Future Improvements

Deploy as a Web Application

Create a Mobile App for farmers

Add real-time camera detection

Expand dataset to include more fruits and vegetables

Use Transfer Learning (ResNet, MobileNet, EfficientNet) for higher accuracy
