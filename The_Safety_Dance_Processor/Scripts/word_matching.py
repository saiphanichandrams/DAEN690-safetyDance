################  Environment Setting  ##################

# CUDA Things, Check CUDA functionality
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set variables
current_dir = os.getcwd()

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
    "caught"
])
# List of irrelevant or common words to ignore
exclusion_list = set([
    "use", "used", "all", "as", "old", "found", "would", "safe", "close", "hard", "arm", "store", "overhead", "pan",
    "strap", "gates", "stick", "secured", "catch", "rack", "function", "later", "sound", "turning", "kid", 
    "secure", "frame", "functional", "section", "assembled", "locations", "bent", "car", "sure", "stable", "action",
    "cooking", "unable", "cute"
])

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

csv_file_path = 'checkpoint3.csv'
final_columns = [
    'rating', 'review_title', 'text', 'parent_asin', 'review_timestamp', 
    'sentiment_score_raw', 'sentiment_score', 'cleaned_text', 'title', 
    'item_model_number', 'manufacturer', 'brand', 'UPC', 'involved_in_incident', 
    'incident_report_date', 'incident_description', 
    'incident_purchase_date']


# Define data types for columns
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
    'sentiment_score': 'float32'}

try:
    print("Loading CSV in chunks...")
    chunk_size = 1000000
    df_list = []
    for chunk in pd.read_csv(
        csv_file_path, usecols=final_columns, dtype=dtypes,
        chunksize=chunk_size, engine='python'
    ):
        df_list.append(chunk)
    df = pd.concat(df_list, ignore_index=True)
    print("CSV successfully loaded!")
except pd.errors.ParserError as e:
    print(f"Parser error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

print('Preprocessing...')
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