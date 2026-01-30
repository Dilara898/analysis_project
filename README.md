# Sentiment Analysis App

##Task 1 -- Project Overview

This project focuses on building a **Sentiment Analysis application**
using a provided English text dataset. The main goal is to automatically
identify the emotion expressed in a sentence written by the user.

The model classifies each sentence into one of the following six
emotions: - anger - fear - joy - love - sadness - surprise

------------------------------------------------------------------------

## General Objective

To build a sentiment analysis model on a given text dataset, evaluate
its performance, and clearly explain the obtained results.

------------------------------------------------------------------------

## Task 2 -- Dataset Analysis (EDA)

The dataset is divided into three parts: - Training set: 16,000
samples - Validation set: 2,000 samples - Test set: 2,000 samples

During exploratory data analysis: - Dataset dimensions were checked -
Column structure (`text`, `label`) was verified - Distribution of
emotion labels was analyzed

The analysis showed that the dataset is **imbalanced**, meaning some
emotions appear more frequently than others.

------------------------------------------------------------------------

## Task 3 -- Text Processing

Before training the model, the text data was cleaned using basic
preprocessing steps: Converted all text to lowercase - Removed
numbers - Removed punctuation and special characters - Removed extra
spaces

These steps help the model focus only on meaningful words related to
emotions.

------------------------------------------------------------------------

## Task 4 -- Model Building

Text data was converted into numerical features using the TF-IDF
method.
A Logistic Regression model was trained to classify emotions.

The model was: Trained on the training set - Evaluated using
validation and test sets

------------------------------------------------------------------------

## Task 5 -- Results and Interpretation

### Model Performance

-   Validation Accuracy: **86.3%**
-   Test Accuracy: **86.9%**

The close values of validation and test accuracy indicate that the model
generalizes well and does not suffer from overfitting.

### Class-wise Performance

Emotions with more training samples achieved better performance.\
Emotions with fewer samples showed lower recall and F1-scores, which is
expected due to dataset imbalance.

### Macro vs Weighted Average

-   Macro average gives equal importance to all classes
-   Weighted average reflects overall performance more realistically

### Improvement Suggestions

Possible improvements include: - Balancing the dataset using
oversampling or undersampling - Using `class_weight='balanced'` in the
model - Trying more advanced models (SVM, neural networks) - Applying
more advanced text preprocessing techniques

------------------------------------------------------------------------

## Task 6 -- Streamlit Interface

A simple **Streamlit-based interface** was created to test the model
interactively.\
Users can enter a sentence and instantly see the predicted emotion.

### Important Note

The dataset does not include a **neutral** emotion class.\
Therefore, neutral sentences are mapped to the closest emotion.

Example: \> "I did my all homeworks" → anger

This behavior is expected and reflects the dataset limitations.

------------------------------------------------------------------------

## How to Run the Project



1. Clone the repository
```bash
git clone https://github.com/Dilara898/analysis_project.git
cd analysis_project


2.  Install dependencies

``` bash
python -m pip install -r requirements.txt
```

(For Anaconda users)

``` bash
C:\ProgramData\anaconda3\python.exe -m pip install -r requirements.txt
```

3.  Train the model

``` bash
python train.py
```

4.  Run the Streamlit app

``` bash
C:\ProgramData\anaconda3\python.exe -m streamlit run app.py
```

5.  Open the browser at

    http://localhost:8501

------------------------------------------------------------------------

## Technologies Used

-   Python
-   Pandas
-   Scikit-learn
-   TF-IDF
-   Logistic Regression
-   Streamlit
