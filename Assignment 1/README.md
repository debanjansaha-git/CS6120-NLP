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
    │    │
    │    ├── bpe.py
    │    
    ├─── dataset/
    │    |
    │    ├── train/
    │    │   ├── austen-emma.txt
    │    │   ├── blake-poems.txt
    │    │   └── shakespeare-hamlet.txt
    │    │
    │    └── test/
    │    │   ├── austen-emma-test.txt
    │    │   ├── blake-poems-test.txt
    │    │   ├── shakespeare-hamlet-test.txt
    │    |
    │    └── reference_punkt_tokens.json
    │    
    ├─── download_files.py
    ├─── train.py
    ├─── test.ipynb
    ├─── Report.docx
    │    





## Installation
Clone the repository:
    
    git clone https://github.com/your_username/byte_pair_encoding.git
    cd byte_pair_encoding
    
Install the required dependencies:
    
    Windows:
    pip install -r requirements.txt

    Linux/Mac:
    pip3 install -r requirements.txt

## Usage
Train the BPE model:

    from bpe import BytePairEncoder
    bpe = BytePairEncoder()
    bpe.learn_bpe(text, num_merges)

Evaluate the BPE algorithm:

    python evaluate_bpe.py

## Results
The evaluation results of the BPE algorithm compared to NLTK's default tokenizer are as follows:

Accuracy: XX%
Precision: XX
Recall: XX
F1 Score: XX
Jaccard Similarity: XX

## Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests for any improvements or additional features.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
* NLTK (Natural Language Toolkit): https://www.nltk.org/
* OpenAI GPT-2 Paper: Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. arXiv preprint arXiv:1911.03351.