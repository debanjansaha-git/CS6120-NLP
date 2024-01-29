import re
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

class BytePairEncoder:
    def __init__(self):
        self.vocab = None
        self.placeholder_map = {}
        self.merge_history = []

    def learn_bpe(self, text, num_merges):
        """
        Learn Byte Pair Encoding (BPE) tokens from the given text.

        Args:
        - text: Input text to learn BPE from.
        - num_merges: Number of merge operations to perform.

        Returns:
        - vocab: BPE vocabulary
        """
        # Step 1: Initialize vocabulary with words and their counts
        vocab = defaultdict(int)
        for line in list(text.split('.')):
            # Split text into lines and words
            for word in line.split():
                vocab[''.join(list(word))] += 1
        
        # Step 2: Perform merges
        for i in range(num_merges):
            pairs = defaultdict(int)
            # Iterate over each word in the vocabulary
            for word, freq in vocab.items():
                # print(f'Word: {word}, Freq: {freq}')
                for j in range(len(word)-1):
                    pair = word[j], word[j+1]
                    # Count the frequency of each pair
                    pairs[pair] += freq

            # Find the most frequent pair
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            # Combine the most frequent pair into a new token
            vocab[''.join(best)] = vocab[best[0]] + vocab[best[1]]
            self.merge_history.append(best)

            # Merge the most frequent pair in the text
            text = text.replace(' '.join(best), ''.join(best))
        self.vocab = vocab
        return vocab

    def encode(self, text):
        """Encode text using the learned BPE tokens."""
        encoded_text = text
        self.placeholder_map = {}
        placeholder_id = 0

        for token, _ in self.vocab.items():
            # Replace each token with a placeholder
            placeholder = f"<{placeholder_id}>"
            encoded_text = encoded_text.replace(token, placeholder)
            self.placeholder_map[placeholder] = token
            placeholder_id += 1

        return encoded_text

    def decode(self, encoded_text):
        """Decode text using the learned BPE tokens."""
        decoded_text = encoded_text
        for placeholder, token in self.placeholder_map.items():
            # Replace each placeholder with the corresponding token
            decoded_text = decoded_text.replace(placeholder, token)
        return decoded_text

    def plot_vocabulary_evolution(self):
        if not self.merge_history:
            # Check if any byte pair merges were recorded
            print('No byte pair merges recorded')
            return
        
        # Count the frequency of each byte pair merge
        merge_counts = Counter(self.merge_history)
        merges, frequencies = zip(*merge_counts.most_common())

        # Plot the frequency of the most common byte pair merges
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies)
        plt.title("Frequency of Most Common Byte Pair Merges")
        plt.xlabel("Merge Number")
        plt.ylabel("Frequency")
        plt.savefig("dataset/vocab/bpe_vocab_evolution.png")
        plt.show()

    def plot_vocab_size_evolution(self):
        if not self.vocab:
            # Check if any vocabulary was recorded
            print('No vocabulary recorded.')
            return
        
        # Calculate the size of the vocabulary over merge operations
        vocab_sizes = [len(self.vocab) for _ in range(len(self.merge_history))]

        # Plot the evolution of vocabulary size
        plt.figure(figsize=(10, 6))
        plt.plot(vocab_sizes)
        plt.title("Evolution of Vocabulary Size")
        plt.xlabel("Merge Number")
        plt.ylabel("Vocabulary Size")
        plt.savefig("dataset/vocab/bpe_vocab_size_evolution.png")
        plt.show()
