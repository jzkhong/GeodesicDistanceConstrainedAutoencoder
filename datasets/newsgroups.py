import numpy as np
import re
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import torch
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')


def fetch_newsgroups_data(selected_categories=None):
    """
    Fetch the 12-Newsgroups data for the selected categories.
    
    Parameters:
    selected_categories (list): List of categories to fetch data for.
    
    Returns:
    tuple: (text, labels, labels_name)
    """
    if selected_categories is None:
        selected_categories = [
            'comp.graphics',
            'comp.os.ms-windows.misc',
            'comp.sys.ibm.pc.hardware',
            'comp.sys.mac.hardware',
            'sci.crypt',
            'sci.electronics',
            'sci.med',
            'sci.space',
            'talk.politics.guns',
            'talk.politics.mideast',
            'talk.religion.misc',
            'misc.forsale'
        ]

    newsgroups_data = datasets.fetch_20newsgroups(subset='all', categories=selected_categories, remove=('headers', 'footers'), random_state=1)
    return newsgroups_data.data, newsgroups_data.target, newsgroups_data.target_names


def preprocess_text(text, stop_words=ENGLISH_STOP_WORDS):
    """
    Clean and preprocess text by lowercasing, removing stop words, and non-alphabetical characters.
    
    Parameters:
    text (str): The text to preprocess.
    stop_words (set): Set of stop words to remove from the text.
    
    Returns:
    str: Preprocessed text.
    """
    text = re.sub(r'\n', ' ', text.lower())
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(tokens)


def preprocess_and_embed_text(text_data, model_name='all-MiniLM-L6-v2', device='cuda'):
    """
    Preprocess text data and convert it into sentence embeddings.
    
    Parameters:
    text_data (list): List of text data to preprocess and embed.
    model_name (str): Name of the SentenceTransformer model to use for embeddings.
    device (str): Device to use for model inference ('cuda' or 'cpu').
    
    Returns:
    ndarray: Sentence embeddings as numpy array.
    """
    cleaned_text = [preprocess_text(text) for text in text_data]

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(cleaned_text, convert_to_tensor=True, device=device)

    return embeddings.cpu().numpy()


def split_and_scale_data(X, y, test_size=0.1, random_state=7, stratify=True):
    """
    Split the data into train and test sets and apply standard scaling.
    
    Parameters:
    X (ndarray): Features matrix.
    y (ndarray): Labels.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed.
    stratify (bool): Whether to stratify the split based on labels.
    
    Returns:
    tuple: Scaled training and testing data.
    """
    stratify = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def load_newsgroups_data():
    """
    Load, preprocess, and embed 12-Newsgroups data, then split and scale it.
    
    Returns:
    tuple: Scaled training and testing data, along with labels.
    """
    
    text_data, labels, labels_name = fetch_newsgroups_data()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    X_vec = preprocess_and_embed_text(text_data, device=device)
    
    X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale_data(X_vec, labels)
    
    print("Training matrix shape:", X_train_scaled.shape)
    print("Testing matrix shape:", X_test_scaled.shape)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, labels_name

# Example usage
X_train_scaled, X_test_scaled, y_train, y_test, labels_name = load_newsgroups_data()
