🌸 Iris Flower Classification

This project implements a K-Nearest Neighbors (KNN) classifier to predict the species of Iris flowers based on their sepal and petal dimensions. The classification model is evaluated with various performance metrics to ensure high accuracy.


Objectives:
- Load and preprocess the Iris dataset.
- Explore and visualize the data.
- Train a classification model using KNN.
- Evaluate model performance using appropriate techniques.



 📚 Dataset Information
Columns:
  - `sepal_length` – Sepal Length 
  - `sepal_width` – Sepal Width 
  - `petal_length` – Petal Length 
  - `petal_width` – Petal Width
  - `species` – Target variable (Setosa, Versicolor, Virginica)


Technologies Used
- Python
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn


How to Run the Project

Clone the Repository:

git clone https://github.com/username/Iris-Flower-Classification.git

Navigate to the Project Directory:

cd Iris-Flower-Classification

Create and Activate a Virtual Environment

Create a virtual environment:
python3 -m venv env
Activate the environment:
On Windows:
env\Scripts\activate

Install Required Packages:
pip install -r requirements.txt

Run Data Exploration Script:
python src/data_exploration.py

Run Data Preprocessing Script:
python src/data_preprocessing.py

Train the Model:
python src/model_training.py

Evaluate the Model:
python src/model_evaluation.py

Project Workflow:
1.Data Exploration: Visualize and analyze the dataset.
2.Data Preprocessing: Split and save data for training and testing.
3.Model Training: Train a KNN classifier and generate predictions.
4.Model Evaluation: Assess model performance using confusion matrix and classification report.

📊 Evaluation Metrics
Accuracy: Overall model performance.
Confusion Matrix: Analyzes prediction errors.
Classification Report: Includes precision, recall, and F1-score.




