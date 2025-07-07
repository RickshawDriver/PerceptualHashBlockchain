import re
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv  # Import the csv module to handle CSV file reading
import hashlib
import time
import json
from typing import List, Dict


# Initialize encoder with predefined classes
label_encoder = LabelEncoder()
label_encoder.fit(['Real', 'Fake', 'Opinion', 'Partially True'])  # Define all possible classes

def build_dataset(blockchain, by_title=False):
    """
    Build training data from blockchain: (distance between hashes, label)
    computes hamming distance to every other block
    """
    features = []
    targets = []

    for i, block in enumerate(blockchain.chain[1:]):  # Skip genesis
        reference_hash = block.titlehash if by_title else block.perceptual_hash
        for j, compare_block in enumerate(blockchain.chain[1:]):
            if i == j:
                continue  # Don't compare to self
            compare_hash = compare_block.titlehash if by_title else compare_block.perceptual_hash
            dist = hamming_distance(reference_hash, compare_hash)
            features.append([dist])
            targets.append(label_encoder.transform([compare_block.label])[0])
    
    return np.array(features), np.array(targets)


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

class NewsBlock: #represents a single article block on the chain and stores its content/metadata within
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
        self.titlehash: str = titlehash if titlehash else generate_perceptual_hash(self.title)
        self.perceptual_hash: str = perceptual_hash if perceptual_hash else generate_perceptual_hash(self.text)

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


def train_regression_model(similarities, labels):
    """
    Train a multinomial logistic regression model for multi-class problems
    """
    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(similarities, labels)
    return model

def classify_by_closest_match(input_hash, model, blockchain, by_title=False):
    distances = []
    labels = []
    for block in blockchain.chain[1:]:
        ref_hash = block.titlehash if by_title else block.perceptual_hash
        dist = hamming_distance(input_hash, ref_hash)
        distances.append(dist)
        labels.append(block.label)
    """
    finds the closes match in the blockchain and returns the label
    uses the trained model for prediction
    """
    distances = np.array(distances)
    min_index = np.argmin(distances)
    closest_distance = distances[min_index]
    
    if closest_distance == 0:
        return labels[min_index], 0
    
    dist_array = np.array([[closest_distance]])
    predicted_label = model.predict(dist_array)
    decoded_label = label_encoder.inverse_transform(predicted_label)
    return decoded_label[0], closest_distance

""" Code used to test hamming distance for paraphrased analysis
def test_paraphrased_detection(original_text, paraphrased_text):
    orig_hash = generate_perceptual_hash(original_text)
    para_hash = generate_perceptual_hash(paraphrased_text)
    dist = hamming_distance(orig_hash, para_hash)
    similarity = 1 - dist / len(hex_to_binary(orig_hash))  # normalize score
    print(similarity)
    print(dist)
    return orig_hash, para_hash, dist, round(similarity, 3)
"""

if __name__ == "__main__":
    # Initialize the blockchain
    """ 
    test_paraphrased_detection("Trump to meet with long list of leaders in New York next week -White House", "Trump plans to meet with big list of leaders in New York next week -White House")
    """
    blockchain = NewsBlockchain()
    #trains model once
    use_title = False #uses full text by default
    X, y = build_dataset(blockchain, by_title=use_title)
    if len(set(y)) <2:
        print("Not enough diversierty in labels for training of the model")
        exit()

    model = train_regression_model(X, y)
    while True: #define menu options
        choice = input("Welcome to the Fake News Perceptual Hash Blockchain system!\nPlease Choose one from the following:\n1: Verify News by Text\n2: Verify news by Title\n> ")
        match choice:
            case "1":
                x = input("Please input the content of the article: \n")
                x = generate_perceptual_hash(x)
                print(x)
                input_hash = x
                use_title = False
            case "2":
                y = input("Please input the title of the article: \n")
                y = generate_perceptual_hash(y)
                print(y)
                input_hash = y
                use_title = True
            case _:
                print("Invalid choice, please choose 1 or 2")
                continue

        label, dist = classify_by_closest_match(input_hash, model, blockchain, by_title=use_title)
        print(f"News article is predicted as: {label} (Closest hamming distance: {dist})\n")
