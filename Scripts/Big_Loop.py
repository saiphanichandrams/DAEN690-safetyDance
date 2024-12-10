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
from transformers import (pipeline, BertTokenizer, DistilBertTokenizer, RobertaTokenizer, BigBirdTokenizer, BigBirdForSequenceClassification)
import sentencepiece as spm
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# Deep learning, NLP things
import torch, spacy, nltk, ahocorasick, faiss
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
################################## Important Variables For the script ###################################

# Get the set of English words from the NLTK corpus
english_words = set(words.words())
# Define a list of known invalid terms
unknown_terms = ['unknown', 'none', 'n/a', 'na', 'nan']

script_dir = os.path.dirname(os.path.abspath(sys.argv[0])) if '__file__' in globals() else os.getcwd()
incident_file = os.path.join(script_dir, "Incidents_Extracted_model_number_cleaned2.csv")
recall_file = os.path.join(script_dir, "recall_cleaned_manual.csv")

# A focused set of safety-related terms (single words only)
initial_safety_terms = set([
    # General safety terms
    "choking", "hazard", "risk", "danger", "burn", "injury", "damage", "defect", 
    "malfunction", "recall", "safety", "warning", "fire", "shock", "fall", 
    "strangulation", "toxic", "poison", "bleed", "contamination", "suffocation", 
    "drowning", "collapse", "explosion", "harm", "fatal", "death", 
    "severe", "unsafe", "flammable", "combustible", "electrocution", "overheat", 
    "sharp", "break", "crack", "cut", "laceration", "fracture", "collision", 
    "impact", "entrapment", "abrasion", "amputation", "exposure", "corrosion", 
    "degradation", "faulty", "rupture", "leak", "misuse", "obstruction", "compress", 
    "infection", "bleeding", "fire", "shock", "burning", "hot", "collapse", 
    "overload", "stuck", "compress", "bruise", "corrode", "erode", "scald", "fracture",
    
    # Medical and physical conditions
    "burn", "reaction", "rash", "swelling", "infection", "cut", "wound", "blister", 
    "fracture", "dislocation", "concussion", "frostbite", "heatstroke", "hypothermia", 
    "seizure", "unconscious", "coma", "paralysis", "dehydration", "blindness", 
    "asphyxiation", "trauma", "blood", "scar", "toxicity", "coma", "puncture", 
    "bruising", "bleeding", "wound", "scratch", "scald", "tender", "pain", 
    "unresponsive", "chronic", "acute", "dizziness", "nausea", "sore", "irritation",
    
    # Product-related terms
    "defective", "breakage", "flaw", "malfunction", "failure", "loose", 
    "sharp", "missing", "detached", "unstable", "unbalanced", "leakage", "misaligned", 
    "corroded", "rusted", "overheat", "under-inflated", "misassembled", "cracked", 
    "brittle", "frayed", "tangled", "torn", "splintered", "warped", "bent", 
    "inoperative", "nonfunctional", "damaged", "fragile", "unstable", "weak", 
    "collapsing", "overheating", "unsecure",
    
    # Material-related hazards
    "toxic", "lead", "asbestos", "mercury", "phthalate", 
    "formaldehyde", "pesticide", "chemical", "volatile", "vapors", "gases", 
    "fumes", "particles", "fibers", "dust", "radiation", "microwave", "X-ray", 
    "laser", "carbon", "gas", "substance", "leak", "corrosive", "acid", "solvent", 
    "oxidizer", "irritant", "allergen", "mutagen", "carcinogen", "radioactive", 
    "biohazard", "fiber", "mold", "poison", "vapor", "irritation",
    
    # Fire and electrical safety
    "fire", "ignition", "sparking", "short-circuit", "overload", "overheating", 
    "explosive", "combustion", "electrical", "shock", "surge", "burning", "smoke", 
    "flame", "scorch", "fuse", "sparks", "arc", "electrocuted", "surge", "fuse", 
    "shock", "hot", "charred", "sparking", "flame", "surge", "scorch",
    
    # Vehicle safety-related terms
    "crash", "collision", "airbag", "accelerator", "deceleration", 
    "failure", "impact", "overturn", "skid", "traction", "ejection", "hydraulic",
    
    # Child-related safety terms
    "strangulation", "choking", "suffocation", "sharp", "poisoning", 
    "swallowed", "trap", "pinched", "loose", "unsecured", "flimsy", 
    "caught"])
# List of irrelevant or common words to ignore
exclusion_list = set([
    "use", "used", "all", "as", "old", "found", "would", "safe", "close", "hard", "arm", "store", "overhead", "pan",
    "strap", "gates", "stick", "secured", "catch", "rack", "function", "later", "sound", "turning", "kid", 
    "secure", "frame", "functional", "section", "assembled", "locations", "bent", "car", "sure", "stable", "action",
    "cooking", "unable", "cute"])

##################################### Function Definitions #############################################
####### HELPER FUNCTIONS #########
def to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        value = value.lower()
        return value in ['true', '1', 'yes']
    return False

def to_str(value):
    if pd.isna(value):
        return ''
    return str(value)

def to_list_safe(value):
    if isinstance(value, list):
        return value
    elif isinstance(value, str):
        return value.strip('[]').replace("'", "").split(', ') if value.strip() else []
    elif isinstance(value, bool) or pd.isna(value):
        return []
    return [value]

def safe_eval(x):
    if isinstance(x, str) and x.startswith('['):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []
    elif pd.isna(x) or isinstance(x, bool):
        return []
    return x if isinstance(x, list) else [x]

# Wrapper function for parallel processing with tqdm
def parallel_with_progress(func, data, n_jobs, desc="Processing"):
    # Use tqdm to wrap the data and show a progress bar
    results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(func)(d) for d in tqdm(data, desc=desc)
    )
    return results

## Convert unicode timestamps to datetime
def convert_timestamp(ts):
    return datetime.datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d')

### Save a csv with encoding handling
def save_to_csv_with_encoding_handling(df, filename):
    try:
        df.to_csv(filename, index=False, encoding="utf-8", errors="replace")  # Replaces any invalid characters
    except UnicodeEncodeError as e:
        print(f"UnicodeEncodeError: {e}. Attempting to save with a different encoding.")
        df.to_csv(filename, index=False, encoding="utf-8", errors="replace")

# Helper functions        
def remove_invalid_unicode(s):
    return s.encode('utf-8', 'replace').decode('utf-8')        

# Preprocessing function to clean text and create truncated text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags and special characters
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # remove special characters
    # Tokenize, remove stopwords, and lemmatize
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    # Remove duplicate words
    words = list(dict.fromkeys(words))
    # Join back into a single string
    cleaned_text = ' '.join(words)
    # Create truncated text based on the number of rows, not characters (using an arbitrary row split here)
    truncated_text = cleaned_text.split()[:100]  # For example, taking the first 100 words
    return cleaned_text, ' '.join(truncated_text)

def parallel_apply(data, func, n_jobs=-1, chunksize=10000):
    with tqdm(total=len(data), desc="Processing in Parallel") as pbar:
        results = Parallel(n_jobs=n_jobs, backend="multiprocessing", batch_size=chunksize)(
            delayed(func)(d) for d in data
        )
        pbar.update(len(data))  # Update the progress bar at the end of each chunk
    return results

# Basic clean text, remport UTF-8 issues
def clean_text(text):
    # Remove any characters that are not valid UTF-8
    if isinstance(text, str):
        return text.encode('utf-8', 'ignore').decode('utf-8')
    return text

# Remove spaces, dashes, periods from a cell
def remove_spaces_and_dashes(text):
    if isinstance(text, str):
        return text.replace(' ', '').replace('-', '').replace('.', '')
    return text

# More preprocessing functions
def apply_preprocess(chunk, func):
    return chunk.apply(func)

# Parallel preprocessing, specifically for the merge
def parallel_preprocess(column_data, func, batch_size=100000, n_jobs=-1):
    # Divide the data into fixed-size chunks
    num_batches = len(column_data) // batch_size + 1
    chunks = [column_data[i * batch_size: (i + 1) * batch_size] for i in range(num_batches) if not column_data[i * batch_size: (i + 1) * batch_size].empty]
    
    # Process each batch in parallel
    processed_chunks = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
        delayed(apply_preprocess)(chunk, func) for chunk in tqdm(chunks, desc="Processing batches", unit="batch")
    )
    
    return pd.concat(processed_chunks, ignore_index=True)

######################### Important script functions #########################
# Define a refined function to filter out invalid terms (but retain NaN values)
## This is used at after the review dataset and metadata data is merged, then cut out all the garbase model IDs.
def is_invalid_item_model_number(term):
    if pd.isna(term):
        return False  # Don't filter out NaN values; retain them
    # Convert to lowercase
    term_lower = term.lower()
    # Check if the term matches any known invalid terms (like 'unknown', 'none')
    if any(unknown_term in term_lower for unknown_term in unknown_terms):
        return True  # Filter out known invalid terms
    # Allow model numbers that contain both letters and digits (e.g., 'A500', 'BABYVIEW20')
    if any(char.isdigit() for char in term_lower) and any(char.isalpha() for char in term_lower):
        return False  # Valid model number if alphanumeric
    # Filter out purely alphabetic terms that match known English words
    if term.isalpha() and term in english_words:
        return True  # Invalid if it's purely an English word
    # Filter out terms that are exactly 2 characters long, whether alphabetic or numeric
    if len(term) == 2:
        return True  # Filter out all 2-character model numbers
    # Check for specific cases like "Mini" that are not alphanumeric and should be filtered out
    if len(term) < 4 and term.isalpha():  # Filtering short terms that are only alphabetic
        return True  # Filter out short terms like "Mini"
    return False  # Otherwise, it's a valid model number

def apply_with_progress(data, func):
    results = []
    for item in tqdm(data, desc="Processing", leave=True):
        results.append(func(item))
    return results

# Main pre-processing function
def run_processing(df, meta_df):
    print("Converting timestamps for processing")
    # Apply the timestamp conversion with progress bar
    df['review_timestamp'] = apply_with_progress(df['timestamp'], convert_timestamp)
    print("Filtering all Amazon reviews prior to 2011")
    # Filter out reviews before 2011
    df = df[pd.to_datetime(df['review_timestamp']).dt.year >= 2011]
    print("Pre-processing text....")
    # Ensure that text values are strings, replacing NaN or non-string values with an empty string
    df['text'] = df['text'].astype(str).replace(np.nan, '', regex=True)
    # Apply the preprocessing function with progress bar to the filtered dataset
    preprocess_results = apply_with_progress(df['text'], preprocess_text)
    # Unzip the results into separate columns
    df['cleaned_text'], df['truncated_text'] = zip(*preprocess_results)
    print(f"Total rows left after filtering and preprocessing: {df.shape[0]}")
    return df  # Return df for further use

# Function to detect language in a single row
def detect_language(text):
    if not isinstance(text, str) or text.strip() == "":
        return 'unknown'
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'

def detect_language_parallel(df):
    df['language'] = Parallel(n_jobs=mp.cpu_count() - 2)(delayed(detect_language)(text) for text in tqdm(df['truncated_text'], desc="Detecting language"))
    return df

################################## Sentiment Analysis Functions: ##########################################
### Presclassify the sentiment based on ratings.
def classify_based_on_rating(rating):
    if rating in [1, 2]:
        return 'NEGATIVE'
    elif rating == 3:
        return 'NEUTRAL'
    elif rating in [4, 5]:
        return 'POSITIVE'

# Function to truncate text based on character length
def truncate_text_batch(df_chunk, max_length=2016):
    df_chunk['truncated_text'] = df_chunk['text'].str.slice(0, max_length)
    # Count truncated rows
    truncation_flags = df_chunk['text'].str.len() > max_length
    affected_count = truncation_flags.sum()
    return df_chunk, affected_count

# Main function to handle truncation in parallel batches
## Used when running sentiment analysis to speed the process up a bit.
def apply_truncate_text_in_batches(df, max_length=2016, batch_size=10000):
    # Split DataFrame into chunks
    chunks = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
    # Use ThreadPoolExecutor to parallelize batches
    affected_total = 0
    results = []
    with ThreadPoolExecutor() as executor:
        # Create a list of futures with tqdm for progress
        futures = [executor.submit(truncate_text_batch, chunk, max_length) for chunk in chunks]
        for future in tqdm(futures, desc="Truncating Text in Parallel"):
            df_chunk, affected_count = future.result()
            results.append(df_chunk)
            affected_total += affected_count
    # Concatenate results into a single DataFrame
    truncated_df = pd.concat(results, ignore_index=True)
    # Report statistics
    total_count = len(df)
    percentage_affected = (affected_total / total_count) * 100
    print(f"Truncation completed. {affected_total} out of {total_count} rows were truncated.")
    return truncated_df

sid = SentimentIntensityAnalyzer()
# Classify based on rating
def classify_based_on_rating(rating):
    return 'NEGATIVE' if rating in [1, 2] else 'NEUTRAL' if rating == 3 else 'POSITIVE'

# VADER sentiment scoring function
def vader_sentiment_analysis(text):
    if isinstance(text, str) and text.strip():
        score = sid.polarity_scores(text)['compound']
        return score
    return None

# Function to adjust sentiment score based on rating
def adjust_sentiment_score(row):
    sentiment_score = row['sentiment_score_raw']
    rating = row['rating']
    if rating <= 2:  # Ratings 1 and 2
        sentiment_score = min(sentiment_score, -0.5) * 1.5  # Force negative, amplify if already negative
    elif rating == 3:  # Rating 3
        sentiment_score *= 0.5  # Neutral sentiment dampened
    elif rating >= 4:  # Ratings 4 and 5
        sentiment_score = max(sentiment_score, 0.5) * 1.5  # Force positive, amplify if already positive
    
    return sentiment_score

# Process a single batch
def process_batch(batch_df):
    # Apply VADER on review text
    batch_df['sentiment_score_raw'] = batch_df['truncated_text'].map(vader_sentiment_analysis)
    # Adjust sentiment score based on rating
    batch_df['sentiment_score'] = batch_df.apply(adjust_sentiment_score, axis=1)
    return batch_df

# Function to process batches in controlled chunks
def limited_batch_sentiment_analysis(df, batch_size=1000, max_active_batches=6):
    total_records = len(df)
    processed_chunks = []

    # Calculate number of rows to process per loop
    loop_batch_size = batch_size * max_active_batches  # Control how many rows are processed at a time
    num_loops = (total_records + loop_batch_size - 1) // loop_batch_size

    with tqdm(total=num_loops, desc="Processing Sentiment Analysis Loops", unit="loop") as loop_bar:
        for loop_start in range(0, total_records, loop_batch_size):
            loop_end = min(loop_start + loop_batch_size, total_records)
            loop_chunk = df.iloc[loop_start:loop_end].copy()

            # Process sub-batches within this loop
            sub_chunks = []
            with tqdm(total=len(loop_chunk) // batch_size + 1, desc="Processing Batches", leave=False) as batch_bar:
                for batch_start in range(0, len(loop_chunk), batch_size):
                    batch_end = min(batch_start + batch_size, len(loop_chunk))
                    batch_df = loop_chunk.iloc[batch_start:batch_end]

                    # Process batch and append results
                    processed_batch = process_batch(batch_df)
                    sub_chunks.append(processed_batch)
                    batch_bar.update(1)

            # Append processed loop chunk to the final result
            processed_chunks.extend(sub_chunks)
            loop_bar.update(1)

    # Concatenate all processed dataframes into one
    return pd.concat(processed_chunks, ignore_index=True)

def clean_text_in_chunks(df, chunk_size=100000):
    """Apply the clean_text function to a DataFrame in chunks."""
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size].copy()
        chunk = chunk.applymap(lambda x: clean_text(x) if isinstance(x, str) else x)
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

################################## Data merging functions: ###########################################
# Function to inspect the 'details' field more thoroughly
## This is used at the beginning to extract the metadata details from the metadata dataset.
def inspect_details(details):
    if isinstance(details, str):
        try:
            details = ast.literal_eval(details)  # Convert string representation of dictionary to actual dictionary
        except:
            return None
    return details

# Enhanced function to search through nested dictionaries ONE LEVEL DOWN and lists for keys (case-insensitive)
def search_nested(details, target_key):
    if isinstance(details, dict):
        for key, value in details.items():
            if key.strip().lower() == target_key.lower():
                return value
            elif isinstance(value, dict):
                result = search_nested(value, target_key)
                if result is not None:
                    return result
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        result = search_nested(item, target_key)
                        if result is not None:
                            return result
    return None

# Function to extract potential ASINs from a string
def extract_asins(text):
    if pd.isna(text):
        return []
    asin_regex = r'\b[bB][a-zA-Z0-9]{9}\b'
    return re.findall(asin_regex, text)

# Function to extract model numbers and potential UPCs from a comma-separated list
def extract_model_numbers_and_upc(model_numbers):
    if pd.isna(model_numbers):
        return [], [], []
    
    # Extract model numbers and any potential UPC (with 9 or more digits)
    model_numbers_list = [num.strip().lower() for num in model_numbers.split(',') if num.strip()]
    upcs = [num for num in model_numbers_list if re.match(r'\b\d{6,13}(?:[-\s]?\d{2,6})?\b', num)]  # Match if the number has 9+ digits
    
    return model_numbers_list, upcs

def check_upc_substring_match(row, upcs_to_match):
    upc_match = any(upc in row['UPC'] for upc in upcs_to_match if isinstance(row['UPC'], str))
    model_id_match = any(upc in row['item_model_number'] for upc in upcs_to_match if isinstance(row['item_model_number'], str))
    return upc_match or model_id_match

# Function to list available categories and their correct dataset names
def list_categories():
    categories = {
        #"All Beauty": "raw_review_All_Beauty",
        #"Toys and Games": "raw_review_Toys_and_Games",
        #"Cell Phones and Accessories": "raw_review_Cell_Phones_and_Accessories",
        #"Industrial and Scientific": "raw_review_Industrial_and_Scientific",
        #"Gift Cards": "raw_review_Gift_Cards",
        #"Musical Instruments": "raw_review_Musical_Instruments",
        #"Electronics": "raw_review_Electronics",
        #"Handmade Products": "raw_review_Handmade_Products",
        #"Arts Crafts and Sewing": "raw_review_Arts_Crafts_and_Sewing",
        #"Baby Products": "raw_review_Baby_Products",
        #"Health and Household": "raw_review_Health_and_Household",
        #"Office Products": "raw_review_Office_Products",
        #"Digital Music": "raw_review_Digital_Music",
        #"Grocery and Gourmet Food": "raw_review_Grocery_and_Gourmet_Food",
        #"Sports and Outdoors": "raw_review_Sports_and_Outdoors",
        #"Home and Kitchen": "raw_review_Home_and_Kitchen",
        #"Subscription Boxes": "raw_review_Subscription_Boxes",
        #"Tools and Home Improvement": "raw_review_Tools_and_Home_Improvement",
        #"Pet Supplies": "raw_review_Pet_Supplies",
        #"Video Games": "raw_review_Video_Games",
        #"Kindle Store": "raw_review_Kindle_Store",
        #"Clothing Shoes and Jewelry": "raw_review_Clothing_Shoes_and_Jewelry",
        #"Patio Lawn and Garden": "raw_review_Patio_Lawn_and_Garden",
        #"Books": "raw_review_Books",
        #"Automotive": "raw_review_Automotive",
        #"CDs and Vinyl": "raw_review_CDs_and_Vinyl",
        "Beauty and Personal Care": "raw_review_Beauty_and_Personal_Care",
        #"Amazon Fashion": "raw_review_Amazon_Fashion",
        #"Magazine Subscriptions": "raw_review_Magazine_Subscriptions",
        #"Software": "raw_review_Software",
        "Health and Personal Care": "raw_review_Health_and_Personal_Care",
        "Appliances": "raw_review_Appliances",
        #"Movies and TV": "raw_review_Movies_and_TV"
    }
    return categories

# Function to load reviews with a manual progress bar
def load_reviews(dataset_name):
    print(f"\nDownloading raw review data for '{dataset_name}'...")
    # Load the dataset without immediately converting it to DataFrame
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", dataset_name, split="full", trust_remote_code=True).shuffle(seed=42)#.select(range(200000))
    ######################################### SAMPLING THE DATASET!! REMOVE AFTER TESTING!! ###########################
    #sampled_dataset = dataset.shuffle(seed=42).select(range(1000000))
    ######################################### SAMPLING THE DATASET!! REMOVE AFTER TESTING!! ###########################
    # Set up the manual progress bar for iterating through the dataset
    data = []
    with tqdm(total=len(dataset), desc="Downloading Review Data", unit="records") as pbar:     ######################################### SAMPLING THE DATASET!! REMOVE AFTER TESTING!! ###########################
        for record in dataset:
            data.append(record)
            pbar.update(1)  # Update the progress bar for each record
    # Convert the collected records into a DataFrame
    df = pd.DataFrame(data)
    return df

# Function to load metadata with a manual progress bar
def load_metadata(dataset_name):
    try:
        # Load the metadata dataset
        meta_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", dataset_name.replace('raw_review', 'raw_meta'), split="full", trust_remote_code=True).shuffle(seed=42)#.select(range(200000))
        # Set up the manual progress bar for iterating through the metadata
        meta_data = []
        with tqdm(total=len(meta_dataset), desc="Downloading Metadata", unit="records") as pbar:
            for record in meta_dataset:
                meta_data.append(record)
                pbar.update(1)  # Update the progress bar for each record
        # Convert the collected records into a DataFrame
        meta_df = pd.DataFrame(meta_data)
        print(f"Loaded {len(meta_df)} metadata records.")
    except Exception as e:
        print(f"Error while loading metadata: {e}")
        return None
    return meta_df

# UPC and Model ID matching using Aho-Corasick automaton
def find_upc_matches(text):
    if not isinstance(text, str):
        return []
    return [upc for _, upc in A.iter(text)]

###### BIG IMPORTANT ONE: Match ASIN, model numbers, and UPCs against the main dataset. 
def match_asins_model_numbers_and_upc_incident(chunk, incident_df_exploded, incident_info_dict, upc_pattern_dict):
    # Initialize match flags
    chunk['asin_match'] = False
    chunk['model_number_match'] = False
    chunk['upc_match'] = False
    chunk['involved_in_incident'] = False
    # Placeholder columns to avoid KeyError
    chunk['incident_report_date'] = np.nan
    chunk['incident_description'] = np.nan
    chunk['incident_purchase_date'] = np.nan
    chunk['extracted_model_numbers_brand_model'] = np.nan
    chunk['Report Date_upc'] = np.nan  # Placeholder for UPC report date
    chunk['Incident Description_upc'] = np.nan  # Placeholder for UPC incident description
    chunk['Purchase Date_upc'] = np.nan  # Placeholder for UPC purchase date

    # Merge based on parent ASIN
    chunk = pd.merge(chunk, incident_df_exploded[['extracted_asins', 'Report Date', 'Incident Description', 'Purchase Date','Manufacturer / Importer / Private Labeler Name',
    'Brand']],
                     how='left', left_on='parent_asin', right_on='extracted_asins', suffixes=('', '_asin'))
    chunk.loc[chunk['extracted_asins'].notna(), 'asin_match'] = True
    # Separate model numbers into short (6 or fewer characters) and long (greater than 6 characters)
    short_model_numbers = incident_df_exploded[incident_df_exploded['extracted_model_numbers'].str.len() <= 6]
    long_model_numbers = incident_df_exploded[incident_df_exploded['extracted_model_numbers'].str.len() > 6]
    # Converting necessary columns to lowercase and removing whitespace for consistent merging
    chunk['brand'] = chunk['brand'].str.lower().str.strip()
    chunk['manufacturer'] = chunk['manufacturer'].str.lower().str.strip()
    short_model_numbers['Brand'] = short_model_numbers['Brand'].str.lower().str.strip()
    short_model_numbers['Manufacturer / Importer / Private Labeler Name'] = short_model_numbers['Manufacturer / Importer / Private Labeler Name'].str.lower().str.strip()

    # Only keep rows in both datasets that have non-empty values for brand and manufacturer
    short_with_brand = short_model_numbers[short_model_numbers['Brand'].notna() & (short_model_numbers['Brand'] != '')]
    short_with_manufacturer = short_model_numbers[short_model_numbers['Manufacturer / Importer / Private Labeler Name'].notna() & 
                                                  (short_model_numbers['Manufacturer / Importer / Private Labeler Name'] != '')]

    # Enforce exact brand match in the merge
    brand_merge = pd.merge(
        chunk, short_with_brand[['extracted_model_numbers', 'Report Date', 'Incident Description', 'Purchase Date', 'Brand']],
        how='left',
        left_on=['item_model_number', 'brand'],
        right_on=['extracted_model_numbers', 'Brand'],
        suffixes=('', '_brand'))

    # Enforce exact manufacturer match in the merge
    manufacturer_merge = pd.merge(
        chunk,short_with_manufacturer[['extracted_model_numbers', 'Report Date', 'Incident Description', 'Purchase Date', 
                                     'Manufacturer / Importer / Private Labeler Name']],
        how='left',
        left_on=['item_model_number', 'manufacturer'],
        right_on=['extracted_model_numbers', 'Manufacturer / Importer / Private Labeler Name'],
        suffixes=('', '_manufacturer'))
    # Concatenate both brand and manufacturer merges to include all possible short model number matches
    chunk = pd.concat([brand_merge, manufacturer_merge], ignore_index=True)
    # Set model_number_match flag for any match in short model numbers
    chunk['model_number_match'] = chunk['extracted_model_numbers'].notna()
    # For long model numbers (greater than 6 characters)
    chunk = pd.merge(chunk, long_model_numbers[['extracted_model_numbers', 'Report Date', 'Incident Description', 'Purchase Date']],
                     how='left', left_on='item_model_number', right_on='extracted_model_numbers', suffixes=('', '_model'))
    chunk.loc[chunk['extracted_model_numbers_model'].notna(), 'model_number_match'] = True
    # UPC and Model ID matching using Aho-Corasick automaton
    upc_matches_in_upc = chunk['UPC'].apply(find_upc_matches)

    # Set match flags and populate incident data where UPC matches are found
    for i, upc_match_list in enumerate(upc_matches_in_upc):
        if upc_match_list:  # Check if there's any UPC match
            chunk.loc[i, 'upc_match'] = True
            match_upc = upc_match_list[0]  # Get the first matched UPC
            # Populate incident data columns with the matched incident info
            chunk.loc[i, ['incident_report_date', 'incident_description', 'incident_purchase_date']] = incident_info_dict[match_upc]
    # Track involvement
    chunk['involved_in_incident'] = chunk['asin_match'] | chunk['model_number_match'] | chunk['upc_match']
    # Set 'involved_in_incident' to True if any match is found (ASIN, model number, or UPC)
    chunk['involved_in_incident'] = chunk['asin_match'] | chunk['model_number_match'] | chunk['upc_match']

    # Consolidate final incident data from various sources
    chunk['incident_report_date'] = chunk['incident_report_date'] \
        .combine_first(chunk['Report Date_brand']) \
        .combine_first(chunk['Report Date_manufacturer']) \
        .combine_first(chunk['Report Date']) \
        .combine_first(chunk['Report Date_model']) \
        .combine_first(chunk['Report Date_upc'])
        
    chunk['incident_description'] = chunk['incident_description'] \
        .combine_first(chunk['Incident Description_brand']) \
        .combine_first(chunk['Incident Description_manufacturer']) \
        .combine_first(chunk['Incident Description']) \
        .combine_first(chunk['Incident Description_model']) \
        .combine_first(chunk['Incident Description_upc'])

    chunk['incident_purchase_date'] = chunk['incident_purchase_date'] \
        .combine_first(chunk['Purchase Date_brand']) \
        .combine_first(chunk['Purchase Date_manufacturer']) \
        .combine_first(chunk['Purchase Date']) \
        .combine_first(chunk['Purchase Date_model']) \
        .combine_first(chunk['Purchase Date_upc'])
    # Deduplicate based on the specified columns
    dedupe_columns = ['parent_asin', 'review_title', 'text', 'review_timestamp', 'title', 'extracted_model_numbers', 'incident_description', 'incident_purchase_date', 'incident_report_date']
    chunk = chunk.drop_duplicates(subset=dedupe_columns, keep='first')
    print(f"Total rows involved in an incident in this chunk: {chunk['involved_in_incident'].sum()} (after deduplication).")
    return chunk

############################ Recall Merge Functions ###################################3
# Function to match ASINs, model numbers, and UPCs
def match_asins_model_numbers_and_upc_recall(chunk, recall_df_exploded, recall_info_dict):
    chunk['asin_match'] = False
    chunk['model_number_match'] = False
    chunk['upc_match'] = False
    chunk['involved_in_recall'] = False
    chunk['recall_date'] = np.nan
    chunk['recall_summary'] = np.nan
    chunk['recall_title'] = np.nan
    # Merge based on ASIN
    chunk = pd.merge(chunk, recall_df_exploded[['extracted_asins', 'date', 'summary', 'title']],
                     how='left', left_on='parent_asin', right_on='extracted_asins', suffixes=('', '_asin'))
    chunk.loc[chunk['extracted_asins'].notna(), 'asin_match'] = True
    # Match model numbers
    short_model_numbers = recall_df_exploded[recall_df_exploded['extracted_model_numbers'].str.len() <= 6]
    long_model_numbers = recall_df_exploded[recall_df_exploded['extracted_model_numbers'].str.len() > 6]
    #Cleanliness is godliness
    chunk['brand'] = chunk['brand'].str.lower().str.strip()
    chunk['manufacturer'] = chunk['manufacturer'].str.lower().str.strip()

    # Only keep rows in both datasets that have non-empty values for brand and manufacturer
    short_with_brand = short_model_numbers[short_model_numbers['brand_names'].notna() & (short_model_numbers['brand_names'] != '')]
    short_with_manufacturer = short_model_numbers[short_model_numbers['manufacturer'].notna() & 
                                                  (short_model_numbers['manufacturer'] != '')]
    brand_merge = pd.merge(
        chunk, short_model_numbers[['extracted_model_numbers', 'date', 'summary', 'title', 'brand_names']],
        how='left', left_on=['item_model_number', 'brand'], right_on=['extracted_model_numbers', 'brand_names'],
        suffixes=('', '_model'))
    manufacturer_merge = pd.merge(
        chunk, short_model_numbers[['extracted_model_numbers', 'date', 'summary', 'title', 'manufacturer']],
        how='left', left_on=['item_model_number', 'manufacturer'], right_on=['extracted_model_numbers', 'manufacturer'],
        suffixes=('', '_model'))
    
    # Concatenate both brand and manufacturer merges to include all possible short model number matches
    chunk = pd.concat([brand_merge, manufacturer_merge], ignore_index=True)
    # Set model_number_match flag for any match in short model numbers
    chunk['model_number_match'] = chunk['extracted_model_numbers'].notna()
    
    chunk = pd.merge(
        brand_merge, long_model_numbers[['extracted_model_numbers', 'date', 'summary', 'title']],
        how='left', left_on='item_model_number', right_on='extracted_model_numbers',
        suffixes=('', '_long'))
    
    chunk['model_number_match'] = chunk['extracted_model_numbers'].notna()
    chunk.loc[chunk['extracted_model_numbers_long'].notna(), 'model_number_match'] = True
    
    # UPC matching
    def find_upc_matches(text):
        if not isinstance(text, str):
            return []
        return [upc for _, upc in A_recall.iter(text)]

    upc_matches_in_upc = chunk['UPC'].apply(find_upc_matches)
    for i, upc_match_list in enumerate(upc_matches_in_upc):
        if upc_match_list:
            chunk.loc[i, 'upc_match'] = True
            match_upc = upc_match_list[0]
            chunk.loc[i, ['recall_date', 'recall_summary', 'recall_title']] = recall_info_dict[match_upc]
    # Use conditional checks to ensure the columns exist before accessing them
    chunk['recall_date'] = (
        chunk['recall_date']
        .combine_first(chunk['date'])
        .combine_first(chunk['date_model'])
        .combine_first(chunk['date_long']))

    chunk['recall_summary'] = (
        chunk['recall_summary']
        .combine_first(chunk['summary'])
        .combine_first(chunk['summary_model'])
        .combine_first(chunk['summary_long']))

    chunk['recall_title'] = (
        chunk['recall_title']
        .combine_first(chunk['title'])
        .combine_first(chunk['title_model'])
        .combine_first(chunk['title_long']))
    
    # Set the involved_in_recall flag if there’s any match (ASIN, model number, or UPC)
    chunk['involved_in_recall'] = chunk['asin_match'] | chunk['model_number_match'] | chunk['upc_match']
    # Deduplicate on specified columns
    dedupe_columns = ['parent_asin', 'review_title', 'text', 'review_timestamp', 'extracted_model_numbers', 'recall_summary', 'recall_title']
    chunk = chunk.drop_duplicates(subset=dedupe_columns, keep='first')
    print(f"Total reviews involved in a recall in this chunk: {chunk['involved_in_recall'].sum()} (after deduplication).")
    return chunk

# Function to extract ASINs and UPCs from text
def extract_asins(text):
    if pd.isna(text):
        return []
    asin_regex = r'\b[bB][a-zA-Z0-9]{9}\b'
    return re.findall(asin_regex, text)

def extract_model_numbers_and_upc(model_numbers):
    if pd.isna(model_numbers):
        return [], []
    model_numbers_list = [num.strip().lower() for num in model_numbers.split(',') if num.strip()]
    upcs = [num for num in model_numbers_list if re.match(r'\b\d{8,13}(?:[-\s]?\d{4,6})?\b', num)]
    return model_numbers_list, upcs

def check_upc_substring_match(row, upcs_to_match):
    upc_match = any(upc in row['UPC'] for upc in upcs_to_match if isinstance(row['UPC'], str))
    model_id_match = any(upc in row['item_model_number'] for upc in upcs_to_match if isinstance(row['item_model_number'], str))
    return upc_match or model_id_match

############################ Safety Word Processing Function Definitions ###################################
# Define cleaning functions
def remove_spaces_and_dashes(text):
    if isinstance(text, str):
        return re.sub(r'[^a-zA-Z0-9,]', '', text)
    return text

def clean_text(text):
    if isinstance(text, str):
        return text.encode('utf-8', 'ignore').decode('utf-8')
    return text

# Clean up the incident description column.
def safe_preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return np.nan  # Keep NaN for unprocessed entries
    # Rest of your preprocessing steps
    text = re.sub(r'<[^>]+>', '', text)  # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # remove special characters
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    cleaned_text = ' '.join(dict.fromkeys(words))
    return cleaned_text

# Pre-load synonyms from WordNet for each safety term
def load_safety_synonyms(safety_terms):
    synonyms_dict = {}
    for term in tqdm(safety_terms, desc="Loading refined synonyms"):
        synonyms = set()
        for syn in wn.synsets(term):
            # Filter to safety-related lexical domains
            if syn.lexname() in {'noun.state', 'noun.act', 'adj.all'}:
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name())
        synonyms_dict[term] = synonyms
    return synonyms_dict

# Function to search pre-loaded synonyms in a text
def find_safety_terms_in_text(text, safety_synonyms):
    if pd.isna(text):
        return []
    
    text_words = set(text.lower().split())
    matched_terms = []
    
    for term, synonyms in safety_synonyms.items():
        if text_words & synonyms:  # If there’s an intersection, record the term
            matched_terms.append(term)
    
    return matched_terms

# Function to process DataFrame in parallel batches
def apply_safety_terms_in_batches(df, text_column, safety_synonyms, batch_size=1000):
    chunks = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
    results = []
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for chunk in chunks:
            futures.append(executor.submit(process_chunk, chunk, text_column, safety_synonyms))
        for future in tqdm(futures, desc="Processing Safety Terms in Parallel"):
            results.append(future.result())
    
    return pd.concat(results, ignore_index=True)

# Function to process each chunk
def process_chunk(df_chunk, text_column, safety_synonyms):
    df_chunk[f"{text_column}_safety_terms"] = df_chunk[text_column].apply(
        lambda text: find_safety_terms_in_text(text, safety_synonyms)
    )
    return df_chunk

# Pre-load synonyms
safety_synonyms = load_safety_synonyms(initial_safety_terms)

#################### Descriptive Statistics Calculations: ##################
def calculate_match_rate(row):
    incident_terms = set(row['incident_safety_terms'])
    review_terms = set(row['text_safety_terms'])
    match_count = len(incident_terms.intersection(review_terms))
    return match_count / max(len(incident_terms), 1)

###################################### DONE WITH FUNCTION DEFINITIONS ############################################
##################################################################################################################


######################################## Preprocess the incident dataset. ########################################
print('Pre-processing incident dataset')
# Load up the incident_df. 
incident_df = pd.read_csv(incident_file, encoding='ISO-8859-1', on_bad_lines='skip')
incident_df.columns = incident_df.columns.str.strip()
        
# Filter incident dataset
columns_to_select = [
    'Report Date', 'Product Description', 'Manufacturer / Importer / Private Labeler Name',
    'Brand', 'Extracted Model Number', 'Purchase Date', 'Retailer', 
    'Incident Description']
try:
    incident_df = incident_df[columns_to_select]
except KeyError as e:
    missing_columns = [col for col in columns_to_select if col not in incident_df.columns]
    print(f"Missing columns: {missing_columns}")
    raise
# convert incident_df columns to lowercase, remove spaces and dashes
incident_df['Extracted Model Number'] = incident_df['Extracted Model Number'].apply(remove_spaces_and_dashes)
incident_df = incident_df.apply(lambda col: col.str.lower() if col.dtype == 'object' else col)
incident_df = incident_df.applymap(clean_text)
initial_count = len(incident_df)
# Filter out rows with null values in 'Model Name or Number'
incident_df = incident_df.dropna(subset=['Extracted Model Number'])
# Count after removing nulls
count_after_null_removal = len(incident_df)

# Extract ASINs, model numbers, and UPCs for the incidents dataset
incident_df['extracted_asins'] = incident_df[['Manufacturer / Importer / Private Labeler Name',
                                              'Product Description', 'Incident Description', 'Extracted Model Number']].apply(
    lambda x: extract_asins(' '.join(x.dropna().astype(str))), axis=1)
incident_df['extracted_model_numbers'], incident_df['extracted_upcs'] = zip(*incident_df['Extracted Model Number'].apply(extract_model_numbers_and_upc))

# Explode the ASINs, model numbers, and UPCs to handle each one individually
incident_df_exploded = incident_df.explode('extracted_asins').explode('extracted_model_numbers').explode('extracted_upcs')

# Build Aho-Corasick automaton for efficient substring search of UPCs
A = ahocorasick.Automaton()
for upc in incident_df_exploded['extracted_upcs'].dropna().unique():
    A.add_word(upc, upc)
A.make_automaton()
# Build UPC regex pattern for fast substring matching in chunks
unique_upcs = incident_df_exploded['extracted_upcs'].dropna().unique()
upc_pattern = "|".join(re.escape(upc) for upc in unique_upcs)

######################################## Preprocess the recall dataset. ########################################
# Load recall dataset

print("Preprocessing Recall Dataset.")
recall_df = pd.read_csv(recall_file, encoding='ISO-8859-1', on_bad_lines='skip')
columns_to_select = ['Date', 'model.id', 'Summary', 'brand_names', 'retail_names', 'manufacturer', 'Title']
recall_df = recall_df[columns_to_select]
recall_df.columns = recall_df.columns.str.strip().str.lower()
recall_df = recall_df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
recall_df['model.id'] = recall_df['model.id'].apply(remove_spaces_and_dashes)
recall_df = recall_df.applymap(clean_text)
initial_recall_count = len(recall_df)
# Filter out rows with null values in 'model.id'
recall_df = recall_df.dropna(subset=['model.id'])
count_after_null_removal = len(recall_df)
# Process recall_df to extract ASINs, model numbers, and UPCs
recall_df['extracted_asins'] = recall_df.apply(lambda x: extract_asins(' '.join(x.dropna().astype(str))), axis=1)
recall_df['extracted_model_numbers'], recall_df['extracted_upcs'] = zip(*recall_df['model.id'].apply(extract_model_numbers_and_upc))
recall_df_exploded = recall_df.explode('extracted_asins').explode('extracted_model_numbers').explode('extracted_upcs')
A_recall = ahocorasick.Automaton()
for upc in recall_df_exploded['extracted_upcs'].dropna().unique():
    A_recall.add_word(upc, upc)
A_recall.make_automaton()
# Build UPC regex pattern for fast substring matching in chunks
unique_upcs_recall = recall_df_exploded['extracted_upcs'].dropna().unique()
upc_pattern_recall = "|".join(re.escape(upc) for upc in unique_upcs_recall)
upc_pattern_dict_recall = {
    row['extracted_upcs']: f".*{re.escape(row['extracted_upcs'])}.*" 
    for _, row in recall_df_exploded.dropna(subset=['extracted_upcs']).iterrows()}
        
required_columns_recall = ['extracted_upcs', 'date', 'summary', 'title']
missing_columns = [col for col in required_columns_recall if col not in recall_df_exploded.columns]
if missing_columns:
    raise KeyError(f"Missing columns in recall_df_exploded: {missing_columns}") 

recall_info_dict = {
    upc: (row['date'], row['summary'], row['title'])
    for upc, row in recall_df_exploded.dropna(subset=['extracted_upcs']).set_index('extracted_upcs').iterrows()}

#######################################################################################################################

###################### Main function: The Big Loop. ########################
def run_all_categories():
    # Get the category dictionary
    categories = list_categories()
    # Loop through each category
    for category_name, dataset_name in categories.items():
        print(f"\nProcessing category: {category_name}")

        ################ Step 1: Load reviews and metadata for the current category ###############
        df = load_reviews(dataset_name)
        meta_df = load_metadata(dataset_name)
        
        # Skip if data loading fails
        if df is None or meta_df is None:
            print(f"Skipping {category_name} due to data loading failure.")
            continue

        ############################# Step 2: Preprocess the category  #################################
        #Rename the 'title column because 'title' is also the name of the product in the metadata
        df.rename(columns={'title': 'review_title'}, inplace=True)
        
        # Clean up the timestamp, pre-process and truncate text.
        df = run_processing(df, meta_df)
        # Drop unnecessary columns
        df = df[['rating', 'review_title', 'text', 'parent_asin', 'review_timestamp', 'cleaned_text', 'truncated_text']]
        # More of step 2:
        meta_df_key = meta_df[['parent_asin', 'title', 'main_category', 'details']].copy()
        meta_df_key['details'] = meta_df_key['details'].apply(inspect_details)
        # Apply the function to extract 'item_model_number', 'manufacturer', and 'brand'
        keys_to_extract = ['item model number', 'manufacturer', 'brand', 'UPC']
        extracted_details = meta_df_key['details'].apply(lambda x: {key: search_nested(x, key) for key in keys_to_extract})
        # Create separate columns from the extracted details
        extracted_df = pd.DataFrame(extracted_details.tolist(), index=meta_df_key.index)
        # Rename the columns in extracted_df to avoid spaces
        extracted_df.columns = ['item_model_number', 'manufacturer', 'brand', 'UPC']
        # Concatenate the extracted columns to the meta_df_key DataFrame
        meta_df_key = pd.concat([meta_df_key, extracted_df], axis=1)
        # Drop the 'details' column as it is no longer needed
        meta_df_key = meta_df_key.drop(columns=['details'])
        # IMPORTANT PART: 
        # Merge the datasets together to get the titles and everything with progress
        merged_df = pd.merge(df, meta_df_key, on='parent_asin', how='left')
        # Apply the is_invalid_item_model_number filter to the 'item_model_number' column in merged_df
        filtered_df = merged_df[~merged_df['item_model_number'].apply(is_invalid_item_model_number)]
        # Assign the filtered result back to merged_df
        merged_df = filtered_df
        # Saving checkpoint 1
        save_to_csv_with_encoding_handling(merged_df, "checkpoint1.csv")
        # Cleaning up the merged_df
        merged_df = merged_df.applymap(lambda x: remove_invalid_unicode(str(x)))
        invalid_model_numbers = merged_df[merged_df['item_model_number'] == ''].shape[0]

        ################ Step 3: Language detection and filtering non-English reviews ###############
        original_row_count = len(merged_df)
        merged_df = detect_language_parallel(merged_df)
        merged_df = merged_df[merged_df['language'] == 'en']  # Filter non-English reviews
        # Display the number of rows before and after filtering
        filtered_out_percentage = ((original_row_count - len(merged_df)) / original_row_count) * 100
        print(f"Percentage removed in language filter: {filtered_out_percentage}")

        ############################## Step 4: The Merge, part 1: The Mergening. ##############################
        # Mod data types of the columns
        for col in merged_df.columns:
            if merged_df[col].dtype == 'int64':
                merged_df[col] = merged_df[col].astype('int32')
            elif merged_df[col].dtype == 'float64':
                merged_df[col] = merged_df[col].astype('float32')
    
        # Ensure all relevant columns in merged_df are in lowercase, and removed spaces and dashes
        merged_df['parent_asin'] = merged_df['parent_asin'].astype(str).str.lower()
        merged_df['item_model_number'] = merged_df['item_model_number'].astype(str).str.lower()
        merged_df['item_model_number'] = merged_df['item_model_number'].apply(remove_spaces_and_dashes)
        merged_df['UPC'] = merged_df['UPC'].apply(remove_spaces_and_dashes)
        
        ## IMPORTANT: Deletes checkpoint2 file because this is appended during the merge process.
        csv_file_path = 'checkpoint2.csv'
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)
        print(f"\n\nMemory cleanup and optimization completed. Proceeding with the merging process...\n There are {len(merged_df)} rows to process.")
        
        # Set chunk size
        chunk_size = 1000000
        total_records = len(merged_df)
        
        # Variables to track statistics
        total_asin_matches = 0
        total_model_number_matches = 0
        total_upc_matches = 0
        total_involved_rows = 0

        # Dictionary to map UPCs to their incident data for lookup
        upc_pattern_dict = {
            row['extracted_upcs']: f".*{re.escape(row['extracted_upcs'])}.*" 
            for _, row in incident_df_exploded.dropna(subset=['extracted_upcs']).iterrows()}
        incident_info_dict = {
            upc: (row['Report Date'], row['Incident Description'], row['Purchase Date'])
            for upc, row in incident_df_exploded.dropna(subset=['extracted_upcs']).set_index('extracted_upcs').iterrows()}
        # Process the merged_df in chunks
        for start in range(0, total_records, chunk_size):
            end = min(start + chunk_size, total_records)
            chunk = merged_df.iloc[start:end].copy()
            print(f"Processing records from {start} to {end}...")

            # Perform the optimized ASIN, model number, and UPC matching
            chunk = match_asins_model_numbers_and_upc_incident(chunk, incident_df_exploded, incident_info_dict, upc_pattern_dict)
            # Track ASIN matches
            asin_matches = chunk['asin_match'].sum()
            total_asin_matches += asin_matches
            # Track model number matches
            model_number_matches = chunk['model_number_match'].sum()
            total_model_number_matches += model_number_matches
            # Track UPC matches
            upc_matches = chunk['upc_match'].sum()
            total_upc_matches += upc_matches
            # Track total rows involved in an incident
            involved_rows = chunk['involved_in_incident'].sum()
            total_involved_rows += involved_rows
            # Save the processed chunk to a CSV file (retain all records)
            chunk.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=False)
            # Perform garbage collection to free memory
            gc.collect()

        # Now calculate summary statistics
        total_affected_records = total_involved_rows
        total_percentage = (total_affected_records / total_records) * 100

        # Print detailed statistics
        print(f'--- Merge Complete. Incident Dataset Merge: {category_name.replace(" ", "_")} Summary Statistics ---')
        print(f"Total affected records (involved in an incident): {total_affected_records}")
        print(f"Percentage of records with incidents: {total_percentage:.2f}%")
        print(f"Total ASIN matches: {total_asin_matches}")
        print(f"Total model number matches: {total_model_number_matches}")
        print(f"Total UPC matches: {total_upc_matches}")

        # Reload the big beast you just saved into the checkpoint2.csv file
        final_columns = ['rating', 'review_title', 'text', 'parent_asin', 'review_timestamp',
        'cleaned_text', 'title', 'item_model_number', 'manufacturer', 'brand',
        'UPC', 'involved_in_incident', 'incident_report_date', 'incident_description', 
        'incident_purchase_date', 'Manufacturer / Importer / Private Labeler Name',
        'Brand']
        csv_file_path = 'checkpoint2.csv'

        dtypes = {
            'rating': 'float32',
            'review_title': 'string',
            'text': 'string',
            'parent_asin': 'string',
            'review_timestamp': 'string',
            'cleaned_text': 'string',
            'title': 'string',
            'item_model_number': 'string',
            'manufacturer': 'string',
            'brand': 'string',
            'UPC': 'string',
            'involved_in_incident': 'bool',
            'incident_report_date': 'string',
            'incident_description': 'string',
            'incident_purchase_date': 'string',
            'Manufacturer / Importer / Private Labeler Name': 'string',
            'Brand': 'string'}

        try:
            print("Reloading dataset post merge (in chunks...)")
            chunk_size = 1000000
            df_list = []
            for chunk in pd.read_csv(
                csv_file_path, usecols=final_columns, dtype=dtypes,
                chunksize=chunk_size, engine='python'):
                df_list.append(chunk)
            df = pd.concat(df_list, ignore_index=True)
        except pd.errors.ParserError as e:
            print(f"Parser error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
            df = pd.read_csv(csv_file_path, usecols=final_columns, engine='python')
                
        unique_involved_products = df[df['involved_in_incident'] == True]['parent_asin'].nunique()
        print(f"Total unique products involved in incidents: {unique_involved_products} for {category_name}")

        # Filter for rows where involved_in_incident is True
        df_involved = df
        if 'involved_in_incident' not in df.columns:
            df['involved_in_incident'] = False #Prevent errors in case no incidents occured.
        df_involved = df_involved[df_involved['involved_in_incident'] == True]
        # Save this filtered dataframe to a separate CSV
        df_involved.to_csv(f'incident_involved_records{category_name.replace(" ", "_")}.csv', index=False)
        
        ############################## Step 5: Sentiment Analysis ##############################
        # Truncate the text again, this time a different size
        df = apply_truncate_text_in_batches(df, max_length=2016, batch_size=10000)
        
        # Initializing VADER. I am your Father.
        sid = SentimentIntensityAnalyzer()
        tqdm.pandas(desc="Running VADER Sentiment Analysis on review text:")
        #First run against the review text
        df = limited_batch_sentiment_analysis(df, batch_size=5000, max_active_batches=6)
        # Adjust the sentiment score based on rating
        df['sentiment_score'] = df['sentiment_score_raw']
        df.to_csv("checkpoint3.csv", index=False, encoding='utf-8')
        
        ############################## Step 6: Safety words: The Safety Dance ##############################
        #Load up the large spacy model now.
        nlp = spacy.load("en_core_web_lg")
        # Ensure both 'incident_description' and 'text' fields have valid data and convert to lowercase
        df['incident_description'] = df['incident_description'].str.lower()
        df['text'] = df['text'].str.lower()
        df['cleaned_incident_description'] = df['incident_description'].apply(safe_preprocess_text)
        df['cleaned_text'] = df['text'].apply(safe_preprocess_text)
        print('Word matching against the incident_description column')
        df = apply_safety_terms_in_batches(df, "incident_description", safety_synonyms)
        print('Word matching against the review text column')
        df = apply_safety_terms_in_batches(df, "text", safety_synonyms)
        
        # Save to CSV, handling encoding issues
        df.to_csv("checkpoint4.csv", index=False, encoding="utf-8", errors="replace")
        
        ############################## Step 7: The Merge, part 2 (Recall): Revenge of the Merge ##############################
        # Check if checkpoint5.csv exists, and delete it if so to avoid duplication
        if os.path.exists("checkpoint5.csv"):
            os.remove("checkpoint5.csv")
            print("Existing checkpoint5.csv file removed. Starting fresh...")
        print(f'Beginning recall merge for {category_name.replace(" ", "_")}.')
        # Initialize counters for statistics
        total_asin_matches = 0
        total_model_number_matches = 0
        total_upc_matches = 0
        total_involved_rows = 0
        unique_recalled_products = set()
        chunk_size = 1000000
        total_records = len(df)
        # Build the dictionary if all required columns are present
        recall_info_dict = {
            upc: (row['date'], row['summary'], row['title'])
            for upc, row in recall_df_exploded.dropna(subset=['extracted_upcs']).set_index('extracted_upcs').iterrows()}

        # The actual recall merge
        for i, start in enumerate(tqdm(range(0, total_records, chunk_size), desc="Processing Batches"), 1):
            end = min(start + chunk_size, total_records)
            chunk = df.iloc[start:end].copy()
            # Perform the matching
            chunk = match_asins_model_numbers_and_upc_recall(chunk, recall_df_exploded, recall_info_dict)

            # Update statistics
            total_asin_matches += chunk['asin_match'].sum()
            total_model_number_matches += chunk['model_number_match'].sum()
            total_upc_matches += chunk['upc_match'].sum()
            total_involved_rows += chunk['involved_in_recall'].sum()
            # Track unique products involved in recalls
            unique_recalled_products.update(chunk.loc[chunk['involved_in_recall'], 'parent_asin'].unique())
            # Append processed chunk to final CSV
            chunk.to_csv("checkpoint5.csv", mode='a', header=not os.path.exists("checkpoint5.csv"), index=False)
            # Cleanup
            gc.collect()
        
        #Prevent errors in case no recalls occured.
        if 'involved_in_recall' not in df.columns:
            df['involved_in_recall'] = False
            
        total_percentage = (total_involved_rows / total_records) * 100
        unique_recalled_count = len(unique_recalled_products)
        print(f'--- Merge Complete. Recall Dataset Merge: {category_name.replace(" ", "_")} Summary Statistics ---')
        print(f"Percentage of records with recalls: {total_percentage:.2f}%")
        print(f"Total ASIN matches: {total_asin_matches}")
        print(f"Total model number matches: {total_model_number_matches}")
        print(f"Total UPC matches: {total_upc_matches}")
        print(f"Total unique products involved in a recall: {unique_recalled_count}")
        # Reload the data post-merge
        csv_file_path = 'checkpoint5.csv'
        use_columns = [
            'rating', 'text', 'parent_asin', 'review_timestamp', 'title', 'review_title',
            'item_model_number', 'involved_in_incident', 'sentiment_score_raw',
            'sentiment_score', 'recall_summary', 'incident_description_safety_terms',
            'text_safety_terms', 'involved_in_recall']
        dtypes = {
            'rating': 'float32',
            'text': 'string',
            'parent_asin': 'object',
            'review_timestamp': 'string',
            'title': 'string',
            'item_model_number': 'string',
            'involved_in_incident': 'bool',
            'sentiment_score_raw': 'float32',
            'sentiment_score': 'float32',
            'recall_summary': 'string',
            'involved_in_recall': 'bool'}
        converters = {
            'incident_desc_sentiment_score': to_float,
            # Uncomment if necessary
            # 'incident_description_safety_terms': to_list_safe,
            # 'text_safety_terms': to_list_safe,
            'involved_in_recall': to_bool}
        # Initialize an empty list to store chunks with involved recalls
        chunks = []
        # Define chunk size
        chunk_size = 1000000  # Adjust based on your system's memory capacity
        try:
            print("Reloading dataset post recalls merge (in chunks...)")
            for chunk in pd.read_csv(
                csv_file_path, usecols=use_columns, dtype=dtypes,
                converters=converters, chunksize=chunk_size, engine='python'):
                # Append each chunk to the list
                chunks.append(chunk)
                print(f"Loaded chunk with {len(chunk)} rows.")
            df = pd.concat(chunks, ignore_index=True)
            print(f"Full dataset loaded with {len(df)} rows.")
        except pd.errors.ParserError as e:
            print(f"Parser error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}. Gonna try to keep going.")
        # Save off the recalls
        recalls_df = df[df['involved_in_recall'] == True]
        recalls_df.to_csv(f'{category_name}_recalls.csv', index=False)
        
        print(f'Filtered dataset with recalls saved as {category_name}_recalls.csv.')
        ########################### Step 8: Aggregation and Heuristic ##########################
        # Start up new dataframe from checkpoint6.csv

        # Some hard stuff. Manage the string lists of the safety words:
        # Convert string representations of lists to actual lists
        df['incident_description_safety_terms'] = df['incident_description_safety_terms'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else ([] if pd.isna(x) else x))
        df['text_safety_terms'] = df['text_safety_terms'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else ([] if pd.isna(x) else x))
        # Aggregate at the product-level data based on parent_asin
        product_df = (
                df.groupby('parent_asin')
                .agg(
                    avg_rating=('rating', 'mean'),
                    review_count=('rating', 'size'),
                    avg_sentiment_score=('sentiment_score', 'mean'),
                    title=('title', 'first'),  # Taking the first title per product
                    item_model_number=('item_model_number', 'first'),  # First item model number per product
                    recall_summary=('recall_summary', 'first'),  # First recall summary per product
                    incident_safety_terms=('incident_description_safety_terms', 
                        lambda x: [term for terms in x if isinstance(terms, list) for term in terms]
                    ),
                    text_safety_terms=('text_safety_terms', 
                        lambda x: [term for terms in x if isinstance(terms, list) for term in terms]
                    ),
                    involved_in_incident=('involved_in_incident', 'max'),  # 1 if any review had an incident
                    involved_in_recall=('involved_in_recall', 'max'),
                    safety_term_count=('text_safety_terms', 
                        lambda x: sum(len(terms) for terms in x if isinstance(terms, list))
                    )).reset_index())
        # Now do some math.
        print(f'Performing heuristic search for {category_name}')
        product_df['safety_term_density'] = product_df['safety_term_count'] / product_df['review_count']
        product_df['incident_review_match_rate'] = product_df.apply(calculate_match_rate, axis=1)
        
        # Set our rules
        rating_threshold = product_df['avg_rating'].quantile(0.25)
        sentiment_threshold = product_df['avg_sentiment_score'].quantile(0.25)
        safety_term_density_threshold = product_df['safety_term_density'].quantile(0.75)  # Higher quantile for stricter filtering
        incident_review_match_threshold = product_df['incident_review_match_rate'].quantile(0.75)              
              
        # Define potentially_unsafe based on combined conditions
        product_df['potentially_unsafe'] = (
            (product_df['avg_rating'] <= rating_threshold) &  # Low rating
            (product_df['avg_sentiment_score'] <= sentiment_threshold) &  # Low sentiment score
            ((product_df['safety_term_density'] >= safety_term_density_threshold) |  # High safety term density OR
            (product_df['incident_review_match_rate'] >= incident_review_match_threshold)) &  # High match rate
            (product_df['involved_in_incident'] == 1) )
        
        # Conversion and saving
        product_df['incident_safety_terms'] = product_df['incident_safety_terms'].apply(
            lambda x: set(eval(x)) if isinstance(x, str) else set())
        product_df['text_safety_terms'] = product_df['text_safety_terms'].apply(
            lambda x: set(eval(x)) if isinstance(x, str) else set())
        product_df.to_csv('checkpoint6.csv', index=False)       

        # Finally, filter down to our final rows.
        unsafe_products_df = product_df[product_df['potentially_unsafe']]
        
        # Save each category’s output CSV
        category_file_name = f'heuristic_unsafe_products_{category_name.replace(" ", "_")}.csv'
        unsafe_products_df.to_csv(category_file_name, index=False)
        print(f'Completed and saved heuristic unsafe for category: {category_name} as {category_file_name}')  

        ############################ Step 9: IsolationForest ######################
        product_df = product_df[product_df['review_count'] >= 15]
        print(f"Filtered to products w/ reviews > 15. Number of products in the filtered DataFrame: {len(df)}")
        # Step 1: Prepare features for the model
        features = [
            'avg_rating', 'review_count', 'avg_sentiment_score',
            'safety_term_density', 'incident_review_match_rate']
        X = product_df[features]
        # Normalize the features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Train the Isolation Forest
        iso_forest = IsolationForest(
            n_estimators=100,  # Number of trees
            contamination=0.0005,  # Proportion of outliers (adjust as needed)
            random_state=42)
        product_df['anomaly_score'] = iso_forest.fit_predict(X_scaled)
        # Flag anomalies
        # Isolation Forest predicts -1 for anomalies and 1 for inliers
        product_df['is_outlier'] = product_df['anomaly_score'] == -1

        # Count flagged outliers
        num_outliers = product_df['is_outlier'].sum()
        print(f"Number of outliers detected: {num_outliers}")

        # Filter flagged anomalies for inspection
        outliers = product_df[product_df['is_outlier']]
        # Save each category’s output CSV
        isolation_file_name = f'isolationforest_unsafe_products_{category_name.replace(" ", "_")}.csv'
        outliers.to_csv(isolation_file_name, index=False)
        print(f'Completed and saved IsolationForest Outliers for category: {category_name} as {category_file_name}')
        # Features to plot
        features_to_plot = [
            'avg_rating', 'review_count', 'safety_term_count',
            'safety_term_density', 'incident_review_match_rate']
        # Ensure no NaN values in 'is_outlier'
        product_df['is_outlier'] = product_df['is_outlier'].fillna(False)
        # Replace inf values with NaN
        product_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Drop rows with NaN in the features_to_plot
        product_df.dropna(subset=features_to_plot, inplace=True)
        # Pairplot with hue for outliers
        pairplot = sns.pairplot(
            product_df,
            vars=features_to_plot,
            hue='is_outlier',  # Highlight outliers
            palette={True: 'red', False: 'blue'},
            diag_kind='kde',
            plot_kws={'alpha': 0.6})
        pairplot.fig.suptitle(f'Pairwise Feature Comparison for {category_name} with IsolationForest Outliers Highlighted', size=16, y=1.02)
        # Save the plot as a JPEG file
        isolation_pairplot = f'isolationforest_pairplot_products_{category_name.replace(" ", "_")}.jpeg'
        pairplot.fig.savefig(isolation_pairplot, format='jpeg', dpi=300)
        print(f'Completed and saved IsolationForest Outliers for category: {category_name} as {category_file_name}')
        
# Call the main loop function to run the entire process
if __name__ == '__main__':
    import multiprocessing as mp
    # Ensure multiprocessing works correctly on Windows
    mp.set_start_method('spawn', force=True)
    run_all_categories()

#############################################################################################################