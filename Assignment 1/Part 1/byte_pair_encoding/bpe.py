from collections import defaultdict
import re

class BytePairEncoder:
    def __init__(self):
        self.vocab = None
        self.placeholder_map = {}

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
                symbols = word.split()
                for j in range(len(symbols)-1):
                    pairs[symbols[j], symbols[j+1]] += freq

            # Find the most frequent pair
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab[''.join(best)] = vocab[best[0]] + vocab[best[1]]

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
