# Byte Pair Encoding (BPE) for Tokenization
This project implements the Byte Pair Encoding (BPE) algorithm for subword tokenization and evaluates its performance compared to standard tokenization methods using NLTK.

## Overview
Tokenization is a critical preprocessing step in natural language processing (NLP), where text is segmented into meaningful units such as words or subwords. Byte Pair Encoding (BPE) is a popular subword tokenization algorithm that learns a vocabulary of variable-length subword units based on the frequency of character sequences.

This project includes the following components:

* Implementation of the BPE algorithm as a Python class.
* Training of the BPE model on the NLTK Gutenberg Corpus.
* Evaluation of the BPE algorithm against NLTK's default tokenizer using accuracy, precision, recall, F1 score, and Jaccard similarity metrics.
* Comparison with NLTK's Punkt tokenizer for reference tokenization.

## Organization

    ├─── byte_pair_encoding/
    │    └── bpe.py
    │    
    ├─── dataset/
    │    |
    │    ├── train/
    │    │   ├── austen-emma.txt
    │    │   ├── blake-poems.txt
    │    │   └── shakespeare-hamlet.txt
    │    │
    │    ├── test/
    │    │   ├── austen-emma-test.txt
    │    │   ├── blake-poems-test.txt
    │    │   └── shakespeare-hamlet-test.txt
    │    |
    │    ├── vocab/
    │    │   ├── bpe_train_vocab.txt
    │    │   ├── reference_punkt_tokens.json
    │    │   ├── bpe_vocab_evolution.png
    │    │   └── bpe_vocab_size_evolution.png
    │    
    ├─── download_files.py
    ├─── train.py
    ├─── test.ipynb
    ├─── Report.pdf
    └─── README.md    



## Installation
Clone the repository:
    
    git clone https://github.com/debanjansaha-git/CS6120-NLP.git
    cd Assignment\ 1/Part\ 1/
    
Install the required dependencies:
    
    Windows:
    pip install -r requirements.txt

    Linux/Mac:
    pip3 install -r requirements.txt

## Usage

The BPE model:

    from bpe import BytePairEncoder
    bpe = BytePairEncoder()
    bpe.learn_bpe(text, num_merges)

Train using the BPE model

    python train.py

BPE algorithm creates a vocabulary and stores it in the folder `dataset/vocab/` \
It also stores the vocabulary evolution and the vocab size evolution in the same folder.

Encode using the BPE algorithm:

    encoded_text = bpe.encode(text)

Decode the encoded text

    decoded_text = bpe.decode(encoded_text)

Evaluate on test (unseen) data

    python test.py
    ipython test.ipynb

This runs the script which uses the learnt vocab from the BPE algorithm and compares it with a baseline version of NLTK's punkt tokenizer. This creates a file and stores it in `dataset/vocab/referemce_punkt_tokens.json`

## Results
The evaluation results of the BPE algorithm compared to NLTK's default tokenizer are as follows:

BPE Algorithm Metrics:

    Accuracy: 100.0%
    Coverage: 100.0%
    Precision: 1.00
    Recall: 1.00
    F1 Score: 1.00
    Jaccard Similarity: 1.00

Default Tokenizer Metrics:

    Accuracy: 0.20%
    Coverage: 0.10%
    Precision: 0.00
    Recall: 0.05
    F1 Score: 0.00
    Jaccard Similarity: 0.00


## Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests for any improvements or additional features.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
* NLTK (Natural Language Toolkit): https://www.nltk.org/
* OpenAI GPT-2 Paper: Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. arXiv preprint arXiv:1911.03351.