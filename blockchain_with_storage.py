import re
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv  # Import the csv module to handle CSV file reading
import hashlib
import time
import json
import imagehash
from PIL import Image
from typing import List, Dict




# Function to clean and normalize text (lowercase, remove punctuation, lemmatize)
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower().strip()

    # Remove punctuation and non-alphabetic characters
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize words
    words = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatize words (reduce to base form)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

def compute_word_frequency(words):
    word_counts = Counter(words)  # Count the frequency of each word
    return word_counts

def generate_perceptual_hash(text):
    # Preprocess the text
    words = preprocess_text(text)

    # Compute word frequency
    word_freq = compute_word_frequency(words)
        
    # Sort word frequency for consistent hash generation
    sorted_freq = sorted(word_freq.items())
        
    # Create a string representation of the sorted frequency list
    freq_str = ''.join([f'{word}:{count}' for word, count in sorted_freq])
        
    # Generate a hash from the frequency string
    perceptual_hash = hashlib.md5(freq_str.encode()).hexdigest()  # Using MD5 for simplicity
        
    return perceptual_hash

class NewsBlock:
    def __init__(self, index: int, previous_hash: str, timestamp: float, 
                 title: str, text: str, subject: str, date: str, label: str, hash_val: str = None, titlehash: str = None,
                 perceptual_hash: str = None, nonce: int = 0) -> None:
        """
        Initialize a new block in the blockchain.
        """
        self.index: int = index
        self.previous_hash: str = previous_hash
        self.timestamp: float = timestamp
        self.title: str = title
        self.text: str = text
        self.subject: str = subject
        self.date: str = date
        self.label: str = label
        self.nonce: int = nonce
        self.hash: str = hash_val if hash_val else self.calculate_hash()
        self.titlehash: str = titlehash if titlehash else generate_perceptual_hash()
        self.perceptual_hash: str = perceptual_hash if perceptual_hash else generate_perceptual_hash()

    def calculate_hash(self) -> str:
        """
        Create a cryptographic hash of the block's essential data using SHA-256.
        """
        block_string: bytes = f"{self.index}{self.previous_hash}{self.timestamp}{self.title}{self.text}{self.subject}{self.date}{self.label}{self.nonce}".encode()
        return hashlib.sha256(block_string).hexdigest()


    def to_dict(self) -> Dict[str, any]:
        """
        Convert block to a dictionary for JSON serialization.
        """
        return {
            'index': self.index,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'title': self.title,
            'text': self.text,
            'subject': self.subject,
            'date': self.date,
            'label': self.label,
            'nonce': self.nonce,
            'hash': self.hash,
            'titlehash': self.titlehash,
            'perceptual_hash': self.perceptual_hash
        }

    @staticmethod
    def from_dict(block_dict: Dict[str, any]) -> 'NewsBlock':
        """
        Create a NewsBlock from a dictionary (used when loading from JSON).
        """
        return NewsBlock(
            index=block_dict['index'],
            previous_hash=block_dict['previous_hash'],
            timestamp=block_dict['timestamp'],
            title=block_dict['title'],
            text=block_dict['text'],
            subject=block_dict['subject'],
            date=block_dict['date'],
            label=block_dict['label'],
            hash_val=block_dict['hash'],
            titlehash=block_dict['titlehash'],
            perceptual_hash=block_dict['perceptual_hash'],
            nonce=block_dict['nonce']
        )

class NewsBlockchain:
    def __init__(self, filename: str = 'blockchain.json') -> None:
        """
        Initialize the blockchain with an optional filename for persistent storage.
        """
        self.filename: str = filename
        self.chain: List[NewsBlock] = self.load_chain_from_file()

    def create_genesis_block(self) -> NewsBlock:
        """
        The first block in the blockchain, known as the Genesis Block.
        """
        return NewsBlock(0, "0", time.time(), "Genesis Block", "This is the genesis block", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A")

    def get_latest_block(self) -> NewsBlock:
        """
        Return the latest block in the blockchain.
        """
        return self.chain[-1]

    def add_block(self, title: str, text: str, subject: str, perceptual_hash: str, date: str, label: str, titlehash: str) -> None:
        """
        Add a new block to the blockchain with the given news article data.
        """
        latest_block: NewsBlock = self.get_latest_block()
        new_block: NewsBlock = NewsBlock(
            index=len(self.chain),
            previous_hash=latest_block.hash,
            timestamp=time.time(),
            title=title,
            text=text,
            subject=subject,
            date=date,
            label=label,
            titlehash=titlehash,
            perceptual_hash=perceptual_hash
        )
        self.chain.append(new_block)
        self.save_chain_to_file()  # Save the updated chain

    def save_chain_to_file(self) -> None:
        """
        Save the blockchain to a JSON file for persistence.
        """
        chain_data: List[Dict[str, any]] = [block.to_dict() for block in self.chain]
        with open(self.filename, 'w') as file:
            json.dump(chain_data, file, indent=4)

    def load_chain_from_file(self) -> List[NewsBlock]:
        """
        Load the blockchain from a JSON file. If the file doesn't exist, create the genesis block.
        """
        try:
            with open(self.filename, 'r') as file:
                chain_data: List[Dict[str, any]] = json.load(file)
                return [NewsBlock.from_dict(block) for block in chain_data]
        except FileNotFoundError:
            # If the file doesn't exist, start with the genesis block
            return [self.create_genesis_block()]

def hex_to_binary(hex_str):
    """
    Convert a hexadecimal string to a binary string.
    """
    # Convert hex to an integer and then to a binary string, stripping the '0b' prefix
    return bin(int(hex_str, 16))[2:].zfill(len(hex_str) * 4)


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Calculate the Hamming distance between two perceptual hashes.
    """
    # Convert hashes to binary
    bin_hash1 = hex_to_binary(hash1)
    bin_hash2 = hex_to_binary(hash2)

    # Ensure both binary strings are the same length
    if len(bin_hash1) != len(bin_hash2):
        raise ValueError("Hashes must be of the same length.")

    # Calculate Hamming distance
    distance = sum(bit1 != bit2 for bit1, bit2 in zip(bin_hash1, bin_hash2))
    return distance


def compare_hashes(input_hash, blockchain):
    """
    Calculate a similarity score between two perceptual hashes.
    The score will be between 0 and 1, where 1 means identical and 0 means completely different.
    """
    similarities = []
    labels = []

    for block in blockchain.chain[1:]:  # Skip the genesis block
        distance = hamming_distance(input_hash, block.perceptual_hash)
        similarities.append(distance)
        labels.append(1 if block.label == 'Real' else 0)  # 'real' -> 1, 'fake' -> 0

    
    return np.array(similarities).reshape(-1, 1), np.array(labels)

def compare_hashes2(input_hash, blockchain):
    """
    Calculate a similarity score between two perceptual hashes.
    The score will be between 0 and 1, where 1 means identical and 0 means completely different.
    """
    similarities = []
    labels = []

    for block in blockchain.chain[1:]:  # Skip the genesis block
        distance = hamming_distance(input_hash, block.titlehash)
        similarities.append(distance)
        labels.append(1 if block.label == 'Real' else 0)  # 'real' -> 1, 'fake' -> 0
    
    return np.array(similarities).reshape(-1, 1), np.array(labels)

def train_regression_model(similarities, labels):
    """
    Train a logistic regression model to classify real vs fake based on similarity.
    """
    model = LogisticRegression()
    model.fit(similarities, labels)
    return model

def classify_input(input_hash, model, blockchain):
    """
    Classify the user input perceptual hash as real or fake.
    """
    similarities, _ = compare_hashes(input_hash, blockchain)
     # Debugging: Print the similarity to see if it matches any of the real news entries exactly (distance = 0)
    print(f"Input hash similarity: {similarities.ravel()}")

    try:
        # Predict probabilities
        prediction_probs = model.predict_proba(similarities)[:, 1]  # Probability of being 'real' (1)
        # Classify as real if the average probability is > 0.5
        return 'Real' if prediction_probs.mean() > 0.5 else 'Fake'
    except NotFittedError:
        raise ValueError("The model is not fitted, likely due to insufficient class diversity.")

if __name__ == "__main__":
    # Initialize the blockchain
    blockchain = NewsBlockchain()

    while True:
        choice = input("Welcome to the Fake News Perceptual Hash Blockchain system!\nPlease Choose one from the following:\n1: Verify News by Title\n2: Verify news by Text\n> ")
        match choice:
            case "1":
                x = input("Please input the content of the article: \n")
                x = generate_perceptual_hash(x)
                print(x)
                score, labels = compare_hashes(x, blockchain)
                model = train_regression_model(score, labels)
                classification = classify_input(x, model, blockchain)
                print(f"The input news article is classified as: {classification}\n")
            case "2":
                y = input("Please input the title of the article: \n")
                y = generate_perceptual_hash(y)
                print(y)
                score, labels = compare_hashes2(y, blockchain)
                model = train_regression_model(score, labels)
                classification = classify_input(y, model, blockchain)
                print(f"The input news article is classified as: {classification}\n")
            case _:
                print("Invalid choice, please choose 1 or 2")
