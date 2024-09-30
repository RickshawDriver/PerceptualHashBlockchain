import hashlib
import re
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import csv  # Import the csv module to handle CSV file reading
from blockchain_with_storage import NewsBlockchain  # Import the NewsBlockchain class from our blockchain file
from blockchain_with_storage import NewsBlock

# Function to generate normal hash using hashlib
def generate_normal_hash(text, algorithm='sha256'):
    hash_function = getattr(hashlib, algorithm)
    return hash_function(text.encode()).hexdigest()

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

# Function to compute word frequency from processed words
def compute_word_frequency(words):
    word_counts = Counter(words)  # Count the frequency of each word
    return word_counts

# Function to generate a perceptual hash based on word frequency
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


def populate_blockchain_from_csv(csv_filename: str) -> None:
    # Initialize the blockchain (this will automatically load the blockchain from the JSON file or create a genesis block)
    blockchain = NewsBlockchain()
    # Read the CSV file
    with open(csv_filename, newline='', encoding='utf-8') as csvfile:
        # DictReader reads the CSV into dictionaries, where each row is a dictionary with column headers as keys
        reader = csv.DictReader(csvfile)
        
        # Iterate through each row in the CSV file
        for row in reader:
            # Extract data from the row using the column names
            title = row['\ufefftitle']
            text = row['text']
            subject = row['subject']
            date = row['date']
            label = row['label']
            titlehash = generate_perceptual_hash(title)
            perceptual_hash = generate_perceptual_hash(text)
             #Add the extracted data as a block to the blockchain
            blockchain.add_block(title=title, text=text, subject=subject, date=date, label=label, titlehash=titlehash, perceptual_hash=perceptual_hash)
    
    # Print a confirmation message once the blockchain has been updated and saved
    print(f"Blockchain populated with data from {csv_filename} and saved to file.")

# The main block of the program
if __name__ == "__main__":
    # Specify the CSV filename. This should point to the CSV file that contains the news articles.
    csv_filename = 'test.csv'
    
    # Call the function to populate the blockchain using data from the CSV file
    populate_blockchain_from_csv(csv_filename)
