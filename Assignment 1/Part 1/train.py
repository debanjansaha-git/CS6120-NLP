import os
from glob import glob
from byte_pair_encoding.bpe import BytePairEncoder
import re
import string
import nltk
from nltk.corpus import stopwords

# Instantiate BytePairEncoder
bpe = BytePairEncoder()
# Directory containing the training books
train_dir = 'dataset/train/'
vocab_dir = 'dataset/vocab/'

# Number of merge operations
num_merges = 1000

def clean_text(text):
    '''
    Cleans the text by converting to lowercase, removing HTML patterns, URLs, and stopwords.

    Args:
    - text: Input text to be cleaned.

    Returns:
    - cleaned_text: Cleaned text.
    '''
    # Converting the text to lowercase
    cleaned_text = text.lower()

    # Removing HTML patterns like <br />
    html_pattern = re.compile('<.*?>')
    cleaned_text = html_pattern.sub(r'', cleaned_text)

    # Removing URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    cleaned_text = url_pattern.sub(r'', cleaned_text)

    # Removing punctuation characters
    cleaned_text = cleaned_text.translate(str.maketrans('', '', string.punctuation))

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [token for token in cleaned_text.split() if token not in stop_words]

    # Joining the tokens back into a string
    cleaned_text = ' '.join(cleaned_tokens)

    return cleaned_text

# Train the BPE algorithm
vocab = {}
for file_name in glob(train_dir + '*.txt'):
    with open(file_name, 'r', encoding='utf-8') as file:
        text = file.read()
        cleaned_text = clean_text(text)
        bpe.learn_bpe(cleaned_text, num_merges)
        vocab.update(bpe.vocab)

# Plot the evolution of vocabulary size
bpe.plot_vocab_size_evolution()

# Plot the frequency of byte pair merges
bpe.plot_vocabulary_evolution()

# Save the vocabulary to a file
vocab_file = 'bpe_train_vocab.txt'
with open(vocab_dir + vocab_file, 'w', encoding='utf-8') as f:
    for token, freq in vocab.items():
        f.write(f"{token}: {freq}\n")

print("BPE vocabulary saved to", vocab_dir + vocab_file)

