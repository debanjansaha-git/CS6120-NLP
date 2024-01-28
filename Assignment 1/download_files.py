import os
import nltk
from nltk.corpus import gutenberg
from byte_pair_encoding.bpe import BytePairEncoder
from urllib.request import urlopen
import lxml
import requests

# Download NLTK packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
nltk.download('gutenberg')

# Path to the directory containing the training and test books
books_dir = 'dataset/'

# Load data from training books
train_books = [
    gutenberg.raw('austen-emma.txt'),
    gutenberg.raw('blake-poems.txt'),
    gutenberg.raw('shakespeare-hamlet.txt')
]

# Training books file names
train_names = [
    'austen-emma.txt',
    'blake-poems.txt',
    'shakespeare-hamlet.txt'
]

# Save training books in train directory
for name, text in zip(train_names, train_books):
    with open(books_dir + 'train/' + name, 'w') as file:
        file.write(text)

# Download testing books
frankenstein_url = "https://gutenberg.org/cache/epub/41445/pg41445.txt"
dracula_url = "https://gutenberg.org/cache/epub/45839/pg45839.txt"
sherlock_url = "https://gutenberg.org/cache/epub/1661/pg1661.txt"
book_urls = [frankenstein_url, dracula_url, sherlock_url]

test_names = [
    'frankenstein.txt', 
    'dracula.txt', 
    'sherlock-holmes.txt'
]

# Save testing books in test directory
for name, url in zip(test_names, book_urls):
    book_raw = urlopen(url).read().decode('utf8')
    with open(books_dir + 'test/' + name, 'w') as file:
        file.write(book_raw)

