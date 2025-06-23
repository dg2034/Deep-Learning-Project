# Deep-Learning-Project

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: DEEP GHEEWALA

*INTERN ID*: CT04DF578

*DOMAIN*: DATA SCIENCE

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

# 🧠 Image Classification using CNN with TensorFlow

This repository contains a deep learning-based image classification project built with TensorFlow and Keras. The project aims to develop a robust Convolutional Neural Network (CNN) capable of identifying patterns in image data and classifying them into predefined categories.

## 📁 Project Structure

```
.
├── task_2.ipynb     # Main notebook containing code for image classification
├── README.md        # Project documentation
├── my_digit.png     # Image of your own hanwritten digit (should be strictly in png format)  
```

## 📌 Objective

The core objective of this project is to develop a machine learning pipeline to classify images using a CNN. It covers everything from importing libraries, preprocessing the data, building and training the CNN model, to evaluating its performance on unseen data.

## 🛠️ Features

* 📦 Data preprocessing using `ImageDataGenerator`
* 🧱 CNN architecture built using `Sequential` API
* ⚙️ Use of layers like `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense`
* 🧪 Model evaluation with accuracy and loss metrics
* 💾 Model saving for future use or deployment

## 🔍 Overview of the Notebook

The notebook (`task_2.ipynb`) includes the following steps:

### 1. 📚 Importing Libraries

Essential packages such as TensorFlow, NumPy, and Matplotlib are imported for model building, data manipulation, and visualization.

### 2. 📁 Data Loading and Preprocessing

* Image data is loaded using `ImageDataGenerator`.
* Images are resized, rescaled (normalization), and augmented (for training).
* Data is divided into training and testing directories.

### 3. 🧠 Model Architecture

The CNN is built using Keras’ Sequential API and consists of:

* Multiple convolutional and pooling layers
* Flattening and fully connected dense layers
* `ReLU` activation for intermediate layers and `softmax` for the output layer

### 4. 📊 Training the Model

* The model is compiled with `categorical_crossentropy` loss and `adam` optimizer.
* It is trained over several epochs with validation on a test dataset.

### 5. 📈 Performance Evaluation

* Accuracy and loss are plotted over epochs to assess performance.
* The model is evaluated on test data to report final accuracy.

### 6. 💾 Model Saving

* The trained model is saved using the `model.save()` function for later use.

## ⚙️ Requirements

You can install the necessary Python packages using:

```bash
pip install -r requirements.txt
```

Typical dependencies include:

* tensorflow
* numpy
* matplotlib
* keras
* scikit-learn

## 🚀 How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/dg2034/Deep-Learning-Project.git
   cd your-repo-name
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your image dataset in the same working directory.

4. Open the notebook and run the cells sequentially:

   ```bash
   jupyter notebook task_2.ipynb
   ```

## ✅ Results

After training, the model achieves high accuracy on the validation set and is capable of classifying unseen images effectively.

## 🙌 Acknowledgements

Thanks to the TensorFlow and Keras communities for providing excellent tools and documentation. This project is built using open-source tools to promote accessible and reproducible machine learning.

## Input Image

![Image](https://github.com/user-attachments/assets/96ad3456-08d7-4e21-9e7f-6817f07782bc)

## Output Image

![Image](https://github.com/user-attachments/assets/b0a697e4-d158-4fd8-9113-09fd51cc2103)
