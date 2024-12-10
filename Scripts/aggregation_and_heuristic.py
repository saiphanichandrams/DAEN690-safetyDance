# Seed langdetect for consistency in results
DetectorFactory.seed = 0
# multiprocessing check
num_cpus = mp.cpu_count()
n_jobs = mp.cpu_count()-2
################################## Important Variables For the script ###################################

def load_category():
    try:
        with open("chosen_category.txt", "r") as f:
            category = f.read().strip()  # Strip any extra whitespace
        return category
    except FileNotFoundError:
        print("Error: 'chosen_category.txt' not found. Ensure the category is selected first.")
        return None
category = load_category()
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

    
#################### Descriptive Statistics Calculations: ##################

def calculate_match_rate(row):
    incident_terms = set(row['incident_safety_terms'])
    review_terms = set(row['text_safety_terms'])
    match_count = len(incident_terms.intersection(review_terms))
    return match_count / max(len(incident_terms), 1)

df = pd.read_csv(
    'checkpoint5.csv',
    usecols=[
        'rating', 'text', 'parent_asin', 'review_timestamp', 'title',
        'item_model_number', 'involved_in_incident',
        'sentiment_score', 'recall_summary','incident_description_safety_terms', 
        'text_safety_terms', 'involved_in_recall'],
    dtype={
        'rating': 'float',
        'parent_asin': 'object',
        'item_model_number': 'str',
        'involved_in_incident': 'bool',
        'sentiment_score_raw': 'float',
        'sentiment_score': 'float',},
    converters={
        'incident_desc_sentiment_score': to_float,
        #'incident_description_safety_terms': to_list_safe,
        #'text_safety_terms': to_list_safe,
        'involved_in_recall': to_bool,})

recalls_df = df[df['involved_in_recall'] == True]
recalls_df.to_csv(f'{category}_recalls.csv', index=False)
print(f'Filtered dataset with recalls saved as {category}_recalls.csv.')
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
print(f'Performing heuristic search')
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
product_df.to_csv('checkpoint6.csv', index=False)      

# Finally, filter down to our final rows.
unsafe_products_df = product_df[product_df['potentially_unsafe']]
        
# Save each category’s output CSV
category_file_name = f'heuristic_unsafe_products_{category}.csv'
unsafe_products_df.to_csv(category_file_name, index=False)
print(f'Completed and saved for {category} as {category_file_name}')  