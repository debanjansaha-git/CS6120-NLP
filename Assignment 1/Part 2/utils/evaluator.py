import numpy as np
import math
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


def calculate_tpr_fpr(confusion_matrix):
    TN, FP, FN, TP = confusion_matrix.ravel()
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return TPR, FPR

def train_evaluate(model_name, classification_function, X_train, y_train, X_test, y_test, results_df, **kwargs):
    # Train and test using TF representation
    predictions_tf = classification_function(X_train, y_train, X_test, 'TF', **kwargs)
    accuracy_tf = accuracy_score(y_test, predictions_tf)
    confusion_matrix_tf = confusion_matrix(y_test, predictions_tf)
    TPR_tf, FPR_tf = calculate_tpr_fpr(confusion_matrix_tf)
    df_tf = pd.DataFrame({
        'Model': [model_name + '_TF'],
        'Accuracy': [accuracy_tf],
        'TPR': [TPR_tf],
        'FPR': [FPR_tf]
    })

    # Train and test using TF-IDF representation
    predictions_tfidf = classification_function(X_train, y_train, X_test, 'TF-IDF', **kwargs)
    accuracy_tfidf = accuracy_score(y_test, predictions_tfidf)
    confusion_matrix_tfidf = confusion_matrix(y_test, predictions_tfidf)
    TPR_tfidf, FPR_tfidf = calculate_tpr_fpr(confusion_matrix_tfidf)
    df_tfidf = pd.DataFrame({
        'Model': [model_name + '_TF-IDF'],
        'Accuracy': [accuracy_tfidf],
        'TPR': [TPR_tfidf],
        'FPR': [FPR_tfidf]
    })

    # Concatenate dataframes and append to results_df
    results_df = pd.concat([results_df, df_tf, df_tfidf], ignore_index=True)
    
    # Print performance metrics
    print(f"Performance metrics for {model_name} using TF representation:")
    print("Accuracy:", accuracy_tf)
    print("TPR:", TPR_tf)
    print("FPR:", FPR_tf)

    print(f"\nPerformance metrics for {model_name} using TF-IDF representation:")
    print("Accuracy:", accuracy_tfidf)
    print("TPR:", TPR_tfidf)
    print("FPR:", FPR_tfidf)

    return results_df