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
            for word in line.split():
                vocab[''.join(list(word))] += 1
        # Step 2: Perform merges
        for i in range(num_merges):
            pairs = defaultdict(int)
            for word, freq in vocab.items():
                # print(f'Word: {word}, Freq: {freq}')
                for j in range(len(word)-1):
                    pair = word[j], word[j+1]
                    pairs[pair] += freq

            # Find the most frequent pair
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab[''.join(best)] = vocab[best[0]] + vocab[best[1]]
            self.merge_history.append(best)

            # Merge the most frequent pair
            text = text.replace(' '.join(best), ''.join(best))
        self.vocab = vocab
        return vocab

    def encode(self, text):
        """Encode text using the learned BPE tokens."""
        encoded_text = text
        self.placeholder_map = {}
        placeholder_id = 0

        for token, _ in self.vocab.items():
            placeholder = f"<{placeholder_id}>"
            encoded_text = encoded_text.replace(token, placeholder)
            self.placeholder_map[placeholder] = token
            placeholder_id += 1

        return encoded_text

    def decode(self, encoded_text):
        """Decode text using the learned BPE tokens."""
        decoded_text = encoded_text
        for placeholder, token in self.placeholder_map.items():
            decoded_text = decoded_text.replace(placeholder, token)
        return decoded_text

    def plot_vocabulary_evolution(self):
        if not self.merge_history:
            print('No byte pair merges recorded')
            return
        
        merge_counts = Counter(self.merge_history)
        # for merge in self.merge_history:
        #     merge_counts[merge] += 1
        
        merges, frequencies = zip(*merge_counts.most_common())

        plt.figure(figsize=(10, 6))
        plt.plot(frequencies)
        plt.title("Frequency of Most Common Byte Pair Merges")
        plt.xlabel("Merge Number")
        plt.ylabel("Frequency")
        plt.savefig("dataset/vocab/bpe_vocab_evolution.png")
        plt.show()

    def plot_vocab_size_evolution(self):
        if not self.vocab:
            print('No vocabulary recorded.')
            return
        
        vocab_sizes = [len(self.vocab) for _ in range(len(self.merge_history))]

        plt.figure(figsize=(10, 6))
        plt.plot(vocab_sizes)
        plt.title("Evolution of Vocabulary Size")
        plt.xlabel("Merge Number")
        plt.ylabel("Vocabulary Size")
        plt.savefig("dataset/vocab/bpe_vocab_size_evolution.png")
        plt.show()
