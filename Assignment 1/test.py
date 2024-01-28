from byte_pair_encoding.bpe import BytePairEncoder
import os
import json
from collections import Counter
import nltk

# Load the trained BPE vocabulary
vocab_file = 'dataset/train/bpe_train_vocab.txt'
bpe = BytePairEncoder()
with open(vocab_file, 'r', encoding='utf-8') as f:
    vocab = {}
    for line in f:
        token, freq = line.strip().split(': ')
        vocab[token] = int(freq)
    bpe.vocab = vocab

# Directory containing the test books
test_dir = 'dataset/test'

# Initialize NLTK's Punkt tokenizer
def_tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
# Store a reference to the test default tokens
ref_test_tokens = {}

# Evaluate on test books
total_tokens = 0
correct_tokens = 0
total_words = 0

for file_name in os.listdir(test_dir):
    with open(os.path.join(test_dir, file_name), 'r', encoding='utf-8') as file:
        text = file.read()
        
        # Tokenize with the default punkt tokenizer
        def_tokenized_text = def_tokenizer.tokenize(text)
        ref_test_tokens[file_name] = def_tokenized_text

        # Encode the text using the trained BPE vocabulary
        encoded_text = bpe.encode(text)

        # Decode the encoded text
        decoded_text = bpe.decode(encoded_text)

        # BPE Tokenization accuracy and coverage
        original_tokens = len(text.split())
        total_tokens += original_tokens
        decoded_tokens = len(decoded_text.split())
        correct_tokens += sum(1 for token in decoded_text.split() if token.isdigit())
        total_words += len(decoded_text.split())

        print(f"File: {file_name}")
        print(f"Original Text:\n{text[:100]}...")
        print(f"Decoded Text:\n{decoded_text[:100]}...")
        print("")

# Save the tokenized results in a structured format
output_file = 'dataset/reference_punkt_tokens.json'
with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(ref_test_tokens, file, indent=4)

print(f"Reference tokenization saved to {output_file}")

# Calculate metrics
original_tokens_set = set(text.split())
decoded_tokens_set = set(decoded_text.split())

TP = len(decoded_tokens_set.intersection(original_tokens_set))
FP = len(decoded_tokens_set - original_tokens_set)
FN = len(original_tokens_set - decoded_tokens_set)

# Calculate precision, recall, F1 score
precision = TP / (TP + FP) if TP + FP > 0 else 0
recall = TP / (TP + FN) if TP + FN > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
# Calculate Jaccard similarity
jaccard_similarity = len(decoded_tokens_set.intersection(original_tokens_set)) / len(decoded_tokens_set.union(original_tokens_set))
# Calculate tokenization accuracy
tokenization_accuracy = correct_tokens / total_tokens * 100
# Calculate tokenization coverage
coverage = (len(decoded_tokens_set) / len(original_tokens_set)) * 100

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
print(f"Jaccard Similarity: {jaccard_similarity:.2f}")
print(f"Tokenization Accuracy: {tokenization_accuracy:.2f}%")
print(f"Coverage: {coverage:.2f}%")
