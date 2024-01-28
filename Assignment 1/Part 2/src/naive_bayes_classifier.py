import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from utils.evaluator import calculate_tpr_fpr, train_evaluate

clean_df = pd.read_csv('data/clean_df.csv')
# split the cleaned data into training and testing sets of 80-20
X_train, X_test, y_train, y_test = train_test_split(clean_df.iloc[:,0], clean_df.iloc[:,1], test_size=0.2, random_state=42)
print(f"Shape of X_train: {X_train.shape} \nShape of X_test: {X_test.shape}")

# Function to train and test Naive Bayes classifier using TF or TF-IDF
def naive_bayes_classification(X_train, y_train, X_test, feature_representation, **kwargs):
    # Extract features from text data
    if feature_representation == 'TF':
        vectorizer = CountVectorizer()
    elif feature_representation == 'TF-IDF':
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError("Invalid feature representation. Choose 'TF' or 'TF-IDF'.")

    train_data = vectorizer.fit_transform(X_train)
    test_data = vectorizer.transform(X_test)
    
    # Train Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(train_data, y_train)
    
    # Test Naive Bayes classifier
    predictions = classifier.predict(test_data)
    
    return predictions

if __name__ == '__main__':
    results_df = pd.DataFrame(columns=["Model", "Accuracy", "TPR", "FPR"])
    results_df = train_evaluate('NaiveBayes', naive_bayes_classification, X_train, y_train, X_test, y_test, results_df)
    print(results_df)