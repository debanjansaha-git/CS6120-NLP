import os
from glob import glob
from byte_pair_encoding.bpe import BytePairEncoder

# Instantiate BytePairEncoder
bpe = BytePairEncoder()
# Directory containing the training books
train_dir = 'dataset/train/'

# Number of merge operations
num_merges = 100

# Train the BPE algorithm
vocab = {}
for file_name in glob(train_dir + '*.txt'):
    with open(file_name, 'r', encoding='utf-8') as file:
        text = file.read()
        bpe.learn_bpe(text, num_merges)
        vocab.update(bpe.vocab)

# Save the vocabulary to a file
vocab_file = 'bpe_train_vocab.txt'
with open(train_dir + vocab_file, 'w', encoding='utf-8') as f:
    for token, freq in vocab.items():
        f.write(f"{token}: {freq}\n")

print("BPE vocabulary saved to", train_dir + vocab_file)
