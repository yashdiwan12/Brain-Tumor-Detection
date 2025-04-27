GLCM + SVM Image Classification
This project demonstrates how to classify images using GLCM (Gray Level Co-occurrence Matrix) features and a Support Vector Machine (SVM) classifier. The goal is to extract texture features from images, train a model, and use it to classify new images into predefined categories.

The project manually computes GLCM and extracts texture features without relying on scikit-image, providing a customized approach to feature extraction and classification.

Introduction
In this project, we extract GLCM (Gray Level Co-occurrence Matrix) features from images to capture texture information. These features are then used to train a Support Vector Machine (SVM) classifier. The classifier learns from the texture information and can predict the class of new, unseen images based on these features.

The GLCM method calculates various texture metrics such as contrast, homogeneity, energy, and correlation, which describe the spatial arrangement of pixel intensities in an image. By using these features, the SVM classifier can separate images into categories based on their texture.

The project does not use scikit-image for GLCM feature extraction. Instead, the GLCM computation is manually implemented, providing a deeper understanding of the process.

Libraries Used
OpenCV (opencv-python)
Used for image processing tasks, such as reading images, resizing them, and converting them to grayscale. It prepares the images for feature extraction by ensuring uniformity and correctness of input data.

Scikit-learn (scikit-learn)
Provides the tools for machine learning, including the Support Vector Machine (SVM) classifier. It is also used for splitting the dataset into training and testing sets using train_test_split. Additionally, it helps evaluate the model performance through metrics such as accuracy, confusion matrix, and classification report.

Matplotlib (matplotlib)
Used to visualize the results of the model training, including plotting the confusion matrix, classification report, and a bar chart showing the correct predictions per class. Visualization helps in better understanding the model's performance.

Dataset Information
The dataset used in this project consists of images organized in folders, where each folder corresponds to a distinct class. The images within each folder are used to extract texture features using the GLCM method. Dataset used: https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection

Preparing Your Dataset
To use your own dataset:

Create a folder structure where each subfolder represents a class (e.g., "class1", "class2").

Place the images belonging to each class inside their respective folders.

Make sure the dataset contains a variety of images for each class to ensure better training of the model.

How the Model is Trained
The following steps outline the process of training and evaluating the model:

1. Feature Extraction using GLCM
The first step in the model is to extract texture features from the images using the GLCM method. The GLCM computes the spatial relationship between pixel intensities in the image, and from this matrix, we derive important features such as:

Contrast

Homogeneity

Energy

Correlation

These features are crucial for distinguishing between different textures in the images, which are later used for classification.

2. Train-Test Split
The dataset is divided into two parts:

Training Set (80%): Used to train the SVM classifier.

Testing Set (20%): Used to evaluate the model's performance.

We use Scikit-learn’s train_test_split function to divide the data.

3. Train the SVM Model
The Support Vector Machine (SVM) classifier is trained on the texture features extracted from the training set. SVM is a powerful classification algorithm that creates a decision boundary to separate different classes based on the features.

4. Model Evaluation
After training, the model is evaluated using the testing set. Evaluation metrics include:

Accuracy: Overall performance of the model.

Confusion Matrix: Shows the performance for each class, comparing predicted labels vs. actual labels.

Classification Report: Displays precision, recall, and F1-score for each class, which helps in understanding how well the model classifies each category.

Results
Upon running the notebook, the following results are generated:

Accuracy: Displays the percentage of correctly classified images out of all the test images.

Confusion Matrix: Shows how well the model has classified each class, with values representing the number of correct and incorrect predictions.

Classification Report: Provides additional details, such as precision, recall, and F1-score for each class, helping to assess the model’s classification performance in a more granular manner.

Bar Chart: Visual representation of the number of correct predictions per class, giving an easy-to-understand overview of the model’s performance across classes.
