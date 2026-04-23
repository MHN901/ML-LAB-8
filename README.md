# ML-LAB-8
***

# K Nearest Neighbors (KNN) Classification Project

## Project Overview
This project demonstrates the implementation of the K Nearest Neighbors (KNN) algorithm using Python and Scikit-Learn. The goal of the project is to classify artificial data into specific target classes. It walks through the entire machine learning pipeline, from data exploratory analysis and preprocessing to finding the optimal `K` value using the Elbow Method and evaluating the final model.

## Dataset
The dataset used in this project is `KNN_Project_Data.csv`. It contains artificial, anonymized feature columns and a target label:
* **Features:** `XVPM`, `GWYH`, `TRAT`, `TLLZ`, `IGGA`, `HYKR`, `EDFS`, `GUUB`, `MGJM`, `JHZC`
* **Target:** `TARGET CLASS` (Binary categorical variable: 0 or 1)

## Libraries and Dependencies
To run this notebook, you will need the following Python libraries installed:
* `pandas` (Data manipulation and analysis)
* `numpy` (Numerical computations)
* `matplotlib` (Data visualization)
* `seaborn` (Statistical data visualization)
* `scikit-learn` (Machine learning algorithms and preprocessing tools)

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
* Loaded the data using pandas.
* Generated a large Seaborn `pairplot` of the dataset with the `hue` set to `TARGET CLASS` to visually explore the relationships and separability between features.

### 2. Data Preprocessing (Standardization)
* Because the KNN classifier predicts the class of a given test observation by identifying the observations that are nearest to it, the scale of the variables matters significantly. 
* Used Scikit-Learn's `StandardScaler` to standardize the feature variables, ensuring that all features contribute equally to the distance calculations.

### 3. Train-Test Split
* Split the standardized data into training and testing sets using `train_test_split` (with a test size of 30% and a random state for reproducibility).

### 4. Initial KNN Model
* Instantiated a `KNeighborsClassifier` with `n_neighbors = 1`.
* Fit the model to the training data and made predictions on the testing set.
* Evaluated the initial model using a `confusion_matrix` and a `classification_report`.

### 5. Choosing an Optimal K Value (The Elbow Method)
* To improve the model, the "Elbow Method" was utilized to select a better `K` value.
* Iterated through `K` values from 1 to 40, training a new model for each, and recording the error rate of the predictions.
* Plotted a line graph of **Error Rate vs. K Value** to visually identify the point where the error rate stabilizes or drops to its minimum.

### 6. Final Model and Evaluation
* Based on the Elbow Method graph, retrained the model using an optimal value of `K = 31`.
* Generated a final `confusion_matrix` and `classification_report`.
* **Results:** The optimized model successfully achieved an accuracy / F1-score of approximately **84%**, demonstrating a solid improvement over the initial `K=1` model.

## How to Run
1. Make sure you have Python and Jupyter Notebook installed.
2. Clone this repository and ensure `KNN_Project_Data.csv` is in the same directory as the notebook.
3. Open `KNN_Assignment_Solved.ipynb` and run the cells sequentially to observe the data pipeline and model evaluation.
