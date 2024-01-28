# Import Libraries
import numpy as np
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
# Download the IMDb movie reviews dataset
nltk.download('movie_reviews')

import warnings
warnings.filterwarnings("ignore")

from nltk.corpus import movie_reviews

# Access the movie reviews and labels
documents = [(movie_reviews.words(fileid), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]

# Shuffle the documents to ensure a balanced distribution of positive and negative reviews
import random
random.shuffle(documents)

# Print the first review and its label
print("Sample Review:", documents[0][0][:10])  # Displaying the first 10 words for brevity
print("Label:", documents[0][1])


def clean_review(review):
    '''
    Input:
        review: a string containing a review.
    Output:
        review_cleaned: a processed review. 
    '''
    # Converting the reviews to lowercase
    review_cleaned = review.lower()

    # Removing HTML patterns like <br />
    html_pattern = re.compile('<.*?>')
    review_cleaned = html_pattern.sub(r'', review_cleaned) 

    # Removing url's if any in the code
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    review_cleaned = url_pattern.sub(r'', review_cleaned)

    # List of stop words
    stop_words = set(stopwords.words('english')) 

    # Tokenizing using NLTK's punkt tokenizer
    tokenwords = word_tokenize(review_cleaned)
    
    # Removing stop words using the list created above
    result = [] 
    for w in tokenwords: 
      if w not in stop_words: 
        result.append(w)
    
    # Lemmatizing the words
    # Performing lemmatization instead of suggested stemming as lemmatization improved the accuracy of the model
    lemmatizer = WordNetLemmatizer()
    review_cleaned = [lemmatizer.lemmatize(w) for w in result]

    # ps = PorterStemmer()
    # review_cleaned = [ps.stem(w) for w in result]

    # Joining the list of words back into a string
    review_cleaned = " ".join(review_cleaned)

    return review_cleaned

# Apply the clean_review function to the entire dataset
cleaned_documents = [(clean_review(' '.join(words)), category) for words, category in documents]
clean_df = pd.DataFrame(cleaned_documents, columns=["review", "sentiment"])
# Save the cleaned dataset for later use
clean_df.to_csv('data/clean_df.csv', index=False)