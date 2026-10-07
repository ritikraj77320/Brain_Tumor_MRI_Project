# 🧠 Brain Tumor MRI Image Classification

A deep learning-based project for classifying brain MRI images into four categories using **EfficientNetB0 Transfer Learning**.

The trained model is integrated with a **Streamlit web application** that allows users to upload an MRI image and receive an AI-assisted classification.

> ⚠️ **Medical Disclaimer:**  
> This project is developed for research and educational purposes only. It is not a medical diagnostic system and should not be used as a substitute for professional medical evaluation, radiological interpretation, or clinical diagnosis.

---

## 📌 Project Overview

Brain tumors are abnormal growths of cells in the brain that may require early detection and proper medical evaluation.

This project applies **Deep Learning and Transfer Learning** to classify brain MRI images into four categories:

- 🧠 Glioma
- 🧠 Meningioma
- 🟢 No Tumor
- 🧠 Pituitary

The project covers the complete machine learning workflow:

1. Data preprocessing
2. Exploratory Data Analysis
3. Custom CNN development
4. EfficientNetB0 Transfer Learning
5. Model evaluation
6. Prediction generation
7. Streamlit web application

---

## 🎯 Objectives

The main objectives of this project are:

- To preprocess and analyze brain MRI images.
- To develop a deep learning model for MRI image classification.
- To implement Transfer Learning using EfficientNetB0.
- To compare and evaluate deep learning models.
- To evaluate model performance using standard classification metrics.
- To generate predictions for unseen MRI images.
- To build an interactive Streamlit application.
- To provide an AI-assisted research and educational prototype.

---

## 🏗️ Project Architecture

```text
                  Brain MRI Image
                        │
                        ▼
                Image Preprocessing
                        │
                        ▼
              EfficientNetB0 Model
                        │
                        ▼
                Transfer Learning
                        │
                        ▼
                 Feature Extraction
                        │
                        ▼
                Classification Layer
                        │
                        ▼
               Four-Class Prediction
                        │
                        ▼
                Streamlit Web App
```
---

##  📁 Project Structure

Brain_Tumor_MRI_Project/
│
├── image_dataset/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
│
├── models/
│   ├── custom_cnn/
│   ├── custom_cnn_final.keras
│   ├── efficientnetb0_best.keras
│   └── efficientnetb0_final.keras
│
├── outputs/
│   ├── plots/
│   ├── class_names.json
│   ├── custom_cnn_classification_report.txt
│   ├── custom_cnn_history.json
│   ├── custom_cnn_metrics.json
│   ├── efficientnetb0_classification_report.txt
│   ├── efficientnetb0_history.json
│   ├── efficientnetb0_metrics.json
│   └── efficientnetb0_predictions.csv
│
├── src/
│   ├── data_analysis.py
│   ├── evaluate_custom_cnn.py
│   ├── evaluate_efficientnet.py
│   ├── preprocessing.py
│   ├── train_custom_cnn.py
│   └── train_efficientnet.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md

## 🧠 Classification Classes
The model performs four-class classification:
Class	Description
Glioma	Brain MRI images belonging to the glioma class
Meningioma	Brain MRI images belonging to the meningioma class
No Tumor	MRI images without a detected tumor class
Pituitary	Brain MRI images belonging to the pituitary class


## 🤖 Models Used

1. Custom CNN
A custom Convolutional Neural Network was developed for brain MRI image classification.
The custom CNN performs:
- Image feature extraction
- Convolution operations
- Feature learning
- Classification into four categories
The trained custom CNN model is stored in:

models/custom_cnn_final.keras

2. EfficientNetB0 Transfer Learning
The main application uses EfficientNetB0 Transfer Learning.
EfficientNetB0 is used as the primary deep learning architecture for extracting meaningful features from MRI images and performing four-class classification.
The model used by the Streamlit application is:
models/efficientnetb0_best.keras

## 📊 Model Performance

The EfficientNetB0 model achieved the following performance on the test dataset:| Metric | Score |
|---|---:|
| Accuracy | **78.46%** |
| Precision | **82.31%** |
| Recall | **78.46%** |
| F1-Score | **78.69%** |

## Performance Summary:
The EfficientNetB0 Transfer Learning model achieved a test accuracy of 78.46% for the four-class brain MRI classification task.

## 🔄 Project Workflow

1. Brain MRI Dataset
          │
          ▼
2. Data Preprocessing
          │
          ▼
3. Image Resizing & Normalization
          │
          ▼
4. EfficientNetB0
          │
          ▼
5. Transfer Learning
          │
          ▼
6. Feature Extraction
          │
          ▼
7. Classification Layer
          │
          ▼
8. Four-Class Prediction
          │
          ▼
9. Model Evaluation
          │
          ▼
10. Streamlit Web Application

## 🖥️ Streamlit Web Application
The project includes an interactive Streamlit web application for AI-assisted MRI image classification.
The application allows users to upload an MRI image and obtain a prediction from the trained EfficientNetB0 model.

### Application Features
- 📤 Upload JPG, JPEG, or PNG MRI images
- 🧠 Four-class brain MRI classification
- ⚡ EfficientNetB0 Transfer Learning
- 📊 Prediction confidence/probability
- 📈 Model performance information
- 🖥️ Simple and interactive interface
- ⚠️ Medical disclaimer
- 🔬 Research and educational use
---

## 🛠️ Technologies Used

The project was developed using the following technologies:

- **Python** – Core programming language
- **TensorFlow / Keras** – Deep learning model development
- **EfficientNetB0** – Transfer learning architecture
- **NumPy** – Numerical computations
- **Pandas** – Data processing and analysis
- **Matplotlib** – Visualization
- **Seaborn** – Statistical visualization
- **Scikit-learn** – Model evaluation and performance metrics
- **Pillow (PIL)** – Image processing
- **Streamlit** – Web application development
- **VS Code** – Development environment
- **Git & GitHub** – Version control and project hosting

---

## 📊 Dataset

The project uses a brain MRI image dataset containing four classes:

| Class | Description |
|---|---|
| Glioma | MRI images containing glioma tumors |
| Meningioma | MRI images containing meningioma tumors |
| No Tumor | MRI images without a detected tumor |
| Pituitary | MRI images containing pituitary tumors |

The dataset is organized into separate directories for each class.

```text
image_dataset/
│
├── glioma/
├── meningioma/
├── no_tumor/
└── pituitary/
```
## Image Preprocessing

Before training and prediction, MRI images undergo preprocessing including:
- Image resizing to 224 × 224 pixels
- RGB conversion
- Numerical array conversion
- Batch dimension expansion
- Model-compatible preprocessing


## 🧠 Deep Learning Approach
Two deep learning approaches were implemented in this project.

### 1. Custom CNN
A custom Convolutional Neural Network was developed from scratch to learn image features and classify MRI images into four categories.
The model performs:
1. Convolution
2. Feature extraction
3. Pooling
4. Feature learning
5. Classification
The trained model is stored at:
models/custom_cnn_final.keras

### 2. EfficientNetB0 Transfer Learning
EfficientNetB0 was used as the primary model for the final application.
Transfer learning allows the model to use previously learned visual features and fine-tune them for brain MRI classification.
The main model used by the Streamlit application is:
models/efficientnetb0_best.keras

## 📈 Model Evaluation
The models were evaluated using standard classification metrics:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report
EfficientNetB0 Performance
Metric	Score
| Metric | Score |
|---|---:|
| Accuracy | **78.46%** |
| Precision | **82.31%** |
| Recall | **78.46%** |
| F1-Score | **78.69%** |


Detailed evaluation results are available inside the:
outputs/

directory.

## 📁 Important Output Files

The outputs directory contains model evaluation and prediction results.
outputs/
│
├── plots/
├── class_names.json
├── custom_cnn_classification_report.txt
├── custom_cnn_history.json
├── custom_cnn_metrics.json
├── efficientnetb0_classification_report.txt
├── efficientnetb0_history.json
├── efficientnetb0_metrics.json
└── efficientnetb0_predictions.csv

These files contain information related to:
- Training history
- Model metrics
- Classification reports
- Predictions
- Visualization plots
- Class mappings

## 💻 Installation

1. Clone the Repository
git clone https://github.com/ritikraj77320/Brain_Tumor_MRI_Project.git

Move into the project directory:
cd Brain_Tumor_MRI_Project

2. Create a Virtual Environment
python -m venv venv

3. Activate the Virtual Environment
Windows
venv\Scripts\activate

Linux / macOS
source venv/bin/activate

4. Install Dependencies
pip install -r requirements.txt

🚀 Running the Streamlit Application
After installing the dependencies, run:
streamlit run app.py

The application will open in your browser.
Usually, it will be available at:
http://localhost:8501


## 🖥️ Streamlit Application

The Streamlit application provides an interactive interface for brain MRI classification.

### Main Features
- 📤 Upload MRI images
- 🧠 Four-class classification
- ⚡ EfficientNetB0 Transfer Learning
- 📊 Prediction confidence
- 📈 Class probability visualization
- 📋 Model performance information
- ⚠️ Medical disclaimer
- 🔬 Research and educational use

## Prediction Workflow
Upload MRI Image
        ↓
Image Preprocessing
        ↓
EfficientNetB0 Model
        ↓
Feature Extraction
        ↓
Classification
        ↓
Prediction Probabilities
        ↓
Final Predicted Class

## ▶️ Training the Models
The training scripts are available inside the src directory.

Data Analysis
python src/data_analysis.py

Data Preprocessing
python src/preprocessing.py

Train Custom CNN
python src/train_custom_cnn.py

Evaluate Custom CNN
python src/evaluate_custom_cnn.py

Train EfficientNetB0
python src/train_efficientnet.py

Evaluate EfficientNetB0
python src/evaluate_efficientnet.py

## 🔬 Research & Educational Purpose

This project demonstrates the application of deep learning and transfer learning techniques to medical image classification.
The project is intended to demonstrate:
- Computer Vision
- Deep Learning
- Transfer Learning
- Image Classification
- Model Evaluation
- Python Programming
- Streamlit Application Development
- End-to-End Machine Learning Workflow

## ⚠️ Limitations
This project has several limitations:
- The model performance depends on the quality and diversity of the training dataset.
- Predictions may be incorrect for MRI images that differ significantly from the training data.
- The model should not be considered a clinical diagnostic system.
- Model confidence does not represent medical certainty.
- Further validation using clinically verified datasets would be required before any real-world medical application.

## 🔮 Future Improvements
Possible future improvements include:
- Increasing the size and diversity of the dataset
- Applying advanced data augmentation techniques
- Improving model accuracy
- Hyperparameter optimization
- Testing additional transfer learning architectures
- Adding Grad-CAM visualizations for model explainability
- Improving the Streamlit interface
- Adding model comparison dashboards
- Deploying the application using Streamlit Cloud or another cloud platform
- Validating the model on independent clinical datasets

### 👨‍💻 Author

Ritik Raj

Computer Science / Artificial Intelligence & Machine Learning
Brain Tumor MRI Image Classification Project

## 
📄 License
This project is intended for educational and research purposes.
You may modify and use the project for learning and academic purposes with appropriate attribution.

## ⭐ Acknowledgement
This project was developed as an academic and research-oriented implementation of deep learning-based brain MRI image classification using TensorFlow, Keras, EfficientNetB0, and Streamlit.
If you find this project useful, consider giving the repository a ⭐ on GitHub.