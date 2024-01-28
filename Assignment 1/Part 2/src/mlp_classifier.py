import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from utils.evaluator import calculate_tpr_fpr, train_evaluate

clean_df = pd.read_csv('data/clean_df.csv')
# split the cleaned data into training and testing sets of 80-20
X_train, X_test, y_train, y_test = train_test_split(clean_df.iloc[:,0], clean_df.iloc[:,1], test_size=0.2, random_state=42)
print(f"Shape of X_train: {X_train.shape} \nShape of X_test: {X_test.shape}")

# Function to train and test Multilayer Perceptron classifier using TF or TF-IDF
def mlp_classifier(X_train, y_train, X_test, feature_representation, **kwargs):
    
    # Extract model parameters
    mlp_arch = kwargs.get("mlp_arch", (100,))
    max_iter = kwargs.get("max_iter", 500)

    # Extract features from text data
    if feature_representation == 'TF':
        vectorizer = CountVectorizer()
    elif feature_representation == 'TF-IDF':
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError("Invalid feature representation. Choose 'TF' or 'TF-IDF'.")

    train_data = vectorizer.fit_transform(X_train)
    test_data = vectorizer.transform(X_test)
    
    # Train MLP classifier
    classifier = MLPClassifier(hidden_layer_sizes=mlp_arch, max_iter=max_iter)
    classifier.fit(train_data, y_train)
    
    # Test MLP classifier
    predictions = classifier.predict(test_data)
    
    return predictions

if __name__ == '__main__':
    results_df = pd.DataFrame(columns=["Model", "Accuracy", "TPR", "FPR"])
    results_df = train_evaluate("MLP_1", mlp_classifier, X_train, y_train, X_test, y_test, results_df, kwargs={"mlp_arch": (100,), "max_iter": 500})
    results_df = train_evaluate("MLP_2", mlp_classifier, X_train, y_train, X_test, y_test, results_df, kwargs={"mlp_arch": (50, 20), "max_iter": 500})
    results_df = train_evaluate("MLP_3", mlp_classifier, X_train, y_train, X_test, y_test, results_df, kwargs={"mlp_arch": (100, 50, 20), "max_iter": 500})
