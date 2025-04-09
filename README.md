# 🫀 Heart Disease Predictor

## Overview

We are using health data to create a machine learning model that can predict whether or not a person has heart disease. The goal is to assist in early detection and prevention by providing an efficient and accessible diagnostic tool.

## Motivation

Heart disease remains one of the leading causes of death worldwide. Early detection can significantly improve outcomes by enabling timely treatment and lifestyle changes. We chose this problem because it allows us to apply data science to a real-world issue with meaningful impact. This project also gave us a chance to strengthen our technical skills in Python, machine learning, and data visualization.

## Tech Stack

- **Languages**: Python, HTML, CSS  
- **Libraries**:
  - pandas  
  - numpy  
  - matplotlib  
  - seaborn  
  - scikit-learn

## Installation & Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/rachelruddy/MAIS202-Final-Project.git
cd MAIS202-Final-Project
```

### 2. Create a virtual environment (optional but recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4. Run the model

Navigate to the `src/heart/` directory and run:

```bash
python modelTestingHeartPred.py
```

## Dataset

We used the [Heart Disease Dataset on Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset), which includes features such as age, cholesterol, resting blood pressure, and more.

## Model

We implemented a **logistic regression** model to predict heart disease presence based on the dataset.

## Results

*Sample visualizations from model evaluation:*

<img width="447" alt="image" src="https://github.com/user-attachments/assets/df4ff829-fd38-4cef-a92b-0eac1f6cccc9" />

![image](https://github.com/user-attachments/assets/80be7ee9-26c2-4197-b32e-adefa200c1c4)
![ConfusionMatrixMAIS](https://github.com/user-attachments/assets/540ae5e0-e22c-4386-ba73-68b6eec0795f)

> *(Note: Be sure to include your actual chart images in a `results/` folder and update the file names above if needed.)*

## Future Work

- Improve data preprocessing (e.g., feature engineering, outlier handling)
- Try alternative models like Random Forest, SVM, or XGBoost
- Hyperparameter tuning to enhance logistic regression performance
- Build a simple front-end interface for user input

## Contributors

- Rachel Ruddy  
- Sierra Smith  
- Maya Novichenok  
- Lilly Gao
