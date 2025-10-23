# Image Classification with CNN

## 📘 Overview
This project demonstrates **image classification** using **Convolutional Neural Networks (CNNs)** in Python. The model is trained to classify images into predefined categories with high accuracy.

## 🧪 Technologies Used
- **Python 3.9+**
- **Libraries**:
  - `TensorFlow` / `Keras` – Building and training the CNN model
  - `NumPy` – Numerical computations
  - `Matplotlib` – Visualizing data and training progress
  - `Pandas` – Dataset handling
  - `Scikit-learn` – Data preprocessing and evaluation metrics

## 🗂 Project Structure
Image-Classifcation/ │ ├── data/                   # Dataset directory │   ├── train/              # Training images │   ├── validation/         # Validation images │   └── test/               # Test images │ ├── notebooks/ │   └── Image_classification.ipynb  # Main notebook │ ├── models/ │   └── cnn_model.h5        # Trained CNN model │ ├── requirements.txt        # Python dependencies └── README.md               # Project documentation
## 🚀 Getting Started

### Prerequisites
Make sure Python 3.6+ is installed. Install dependencies using:


pip install -r requirements.txt
Dataset

Organize your dataset like this:

data/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── validation/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...

Training the Model

To train the CNN:

python train_model.py

Evaluating the Model

Evaluate performance on test data:

python evaluate_model.py





