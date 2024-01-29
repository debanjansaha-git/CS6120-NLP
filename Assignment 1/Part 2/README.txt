# Sentiment Analysis Project
This project aims to perform sentiment analysis on movie reviews using machine learning algorithms. The goal is to classify each review as either positive or negative sentiment based on the text content.

## Overview
Sentiment analysis, also known as opinion mining, is a natural language processing (NLP) task aimed at determining the sentiment or opinion expressed in a piece of text. In this project, we explore the effectiveness of various machine learning algorithms for sentiment analysis, including Naive Bayes, Logistic Regression, and Multilayer Perceptron (MLP), using different feature representations such as Term Frequency (TF) and Term Frequency-Inverse Document Frequency (TF-IDF).

## Project Structure

    ├─── data/
    │    └── clean_df.csv
    │    
    ├─── utils/
    │    └── evaluator.py
    │    
    ├─── src/
    │    ├─── data_preprocessing.py
    │    ├─── logistic_regression.py
    │    ├─── mlp_classifier.py
    │    ├─── naive_bayes_classifier.py
    │    └─── sentiment_analysis.ipynb
    │
    ├─── README.md
    ├─── requirements.txt
    └─── Report.pdf

data: Contains the dataset used for sentiment analysis.
src:
preprocessing.py: Includes functions for data preprocessing such as text cleaning, tokenization, and feature extraction.
classification.py: Implements the classification algorithms (Naive Bayes, Logistic Regression, MLP) and evaluation functions.
visualization.py: Contains code for generating visualizations of classification results.
results: Stores the output results and visualizations generated during the analysis.
README.md: Markdown file providing an overview of the project.



## Setup Instructions
Clone the repository:

    git clone https://github.com/debanjansaha-git/CS6120-NLP.git
    cd Assignment\ 1/Part\ 2/
    
Install the required dependencies:

    pip install -r requirements.txt

Run the data preprocessing scripts:

    python src/data_preprocessing.py

Run the algorithm scripts:

    python src/logistic_regression.py
    python src/mlp_classifier.py
    python src/naive_bayes_classifier.py

Or Run the main notebook which contains all the scripts:

    ipython src/sentiment_analysis.ipynb


## Results
The results of the sentiment analysis experiment, including classification performance metrics and visualizations, are stored in the results directory.

## Conclusion
This project demonstrates the application of machine learning algorithms for sentiment analysis tasks and provides insights into the effectiveness of different classification techniques and feature representations.