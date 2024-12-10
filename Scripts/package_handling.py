# Suppress warnings and set critical log levels
import os, sys, gc, re, math, time, logging, warnings, psutil, datetime, json, ast, multiprocessing, datasets
warnings.filterwarnings('ignore')
from statistics import mode
# Core libraries and utilities
from collections import Counter
from difflib import SequenceMatcher
from tqdm import tqdm
# Data handling and processing
import numpy as np
import pandas as pd
import seaborn as sns

import dask.dataframe as dd
from multiprocessing import Pool, Manager, cpu_count
import multiprocessing as mp # There's a reason multiprocessing is imported twice here. I have no idea why.
from joblib import Parallel, delayed
# Visualization
import matplotlib.pyplot as plt
# Machine learning and NLP libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import langid
from langid import classify

from transformers import (pipeline, BertTokenizer, DistilBertTokenizer, RobertaTokenizer, BigBirdTokenizer, BigBirdForSequenceClassification)
import sentencepiece as spm
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# Deep learning, NLP things
import torch, spacy, nltk, ahocorasick
from torch.cuda.amp import autocast
from transformers import pipeline, RobertaTokenizer
from torch.utils.data import DataLoader, TensorDataset
# NLP processing and datasets
from datasets import load_dataset
datasets.logging.set_verbosity_error()  # Reduce verbosity from datasets package
nltk.download = lambda *args, **kwargs: None  # Overwrite the nltk.download function to silence it
from nltk import ngrams
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from spacy.matcher import Matcher, PhraseMatcher
from fuzzywuzzy import fuzz, process
from sentence_transformers import SentenceTransformer
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

original_working_dir = os.getcwd()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
for lib in ["transformers", "datasets", "nlp", "nltk", "urllib3", "torch"]:
    logging.getLogger(lib).setLevel(logging.CRITICAL)
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
# Explicitly set the NLTK data directory
nltk_data_dir = '/home/jking47/nltk_data'
# Create the NLTK data directory if it doesn't exist
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
# Set the NLTK_DATA environment variable to the desired folder
os.environ['NLTK_DATA'] = nltk_data_dir
# Force download of the required NLTK corpora into the specified directory
nltk_packages = ['stopwords', 'wordnet', 'omw-1.4', 'words']
for pkg in nltk_packages:
    try:
        # Check if the package is already present
        nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        # Force download the missing package
        nltk.download(pkg, download_dir=nltk_data_dir)
# Add the NLTK data directory to NLTK's data path
nltk.data.path.append(nltk_data_dir)
# Reset the environment to the original working directory
os.chdir(original_working_dir)
# List of models to download with retry logic
models_to_check = [
    ("google/bigbird-roberta-base", BigBirdTokenizer, BigBirdForSequenceClassification),
    ("cardiffnlp/twitter-roberta-base-sentiment-latest", None, None)]
print("Package handling complete!")

########################################################################################################
################  Environment Setting  ##################
# CUDA Things, Check CUDA functionality
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
english_words = set(words.words())
# Set variables
current_dir = os.getcwd()

# Model download checks
model_name = "google/bigbird-roberta-base"
tokenizer = BigBirdTokenizer.from_pretrained(model_name)
model = BigBirdForSequenceClassification.from_pretrained(model_name)
sentiment_model = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
language_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    print("Downloading 'en_core_web_lg' model...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
    nlp = spacy.load("en_core_web_lg")
nlp = spacy.load("en_core_web_sm")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    print("Downloading 'WordNet' lexicon...")
    nltk.download("wordnet")    
try:
    nltk.data.find("vader_lexicon")
except LookupError:
    print("Downloading 'Vader' lexicon...")
    nltk.download("vader_lexicon")    
# CUDA check
if device.type == "cuda":
    print("CUDA detected. Using GPU for processing.")
else:
    print("CUDA not detected. Using CPU for processing.")
use_gpu = model.device.type == 'cuda'
# Seed langdetect for consistency in results
DetectorFactory.seed = 0
# multiprocessing check
num_cpus = mp.cpu_count()
n_jobs = mp.cpu_count()-2