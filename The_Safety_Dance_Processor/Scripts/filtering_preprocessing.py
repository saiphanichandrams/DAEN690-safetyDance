# Function to convert a timestamp to a readable format
#if len(df) > 7_000_000:
#        df = df.sample(n=7_000_000, random_state=42)
#original_length=len(df)
def convert_timestamp(ts):
    return datetime.datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d')

# Initialize lemmatizer and stop words once
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text_vectorized(text_series):
    # Convert to lowercase
    text_series = text_series.str.lower()

    # Remove HTML tags and special characters
    text_series = text_series.str.replace(r'<[^>]+>', '', regex=True)  # Remove HTML tags
    text_series = text_series.str.replace(r'[^a-z\s]', '', regex=True)  # Remove special characters

    # Tokenize, remove stopwords, and lemmatize
    def tokenize_and_process(text):
        words = text.split()
        processed_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return list(dict.fromkeys(processed_words))  # Remove duplicates
    
    tqdm.pandas(desc="Tokenizing and processing text")
    processed_text = text_series.progress_apply(tokenize_and_process)

    # Create cleaned text and truncated text
    cleaned_text = processed_text.apply(lambda x: ' '.join(x))
    truncated_text = processed_text.apply(lambda x: ' '.join(x[:100]))

    return cleaned_text, truncated_text

# Function to process data in parallel
def parallel_apply(data, func, n_jobs=-3):
    results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
        delayed(func)(d) for d in tqdm(data, desc="Processing in Parallel", leave=True, position=0)
    )
    return results

# Main processing function
def run_processing(df, meta_df):
    print("Beginning filtering and pre-processing")
    # Vectorized timestamp conversion
    df['review_timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce').dt.strftime('%Y-%m-%d')
    # Filter out reviews before 2011
    df['review_timestamp'] = pd.to_datetime(df['review_timestamp'], errors='coerce')
    df = df[df['review_timestamp'].dt.year >= 2011]

    # Ensure that text values are strings, replacing NaN or non-string values with an empty string
    df['text'] = df['text'].astype(str).replace(np.nan, '', regex=True)

    # Apply the optimized preprocessing function
    df['cleaned_text'], df['truncated_text'] = preprocess_text_vectorized(df['text'])
    print(f"Total rows left after filtering and preprocessing: {df.shape[0]}")

    return df


# Seed langdetect for consistency in results
DetectorFactory.seed = 0

def detect_language_batch(texts):
    results = []
    for text in texts:
        if not isinstance(text, str) or text.strip() == "":
            results.append("unknown")
        else:
            results.append(classify(text)[0])  # Get language code from classify
    return results

# Batch processing with langid
def detect_language_langid_batch(df, batch_size=1000):
    # Ensure 'truncated_text' contains only strings
    df['truncated_text'] = df['truncated_text'].astype(str).fillna('')
    
    # Split into batches
    batches = [df['truncated_text'].iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]
    
    # Process each batch and collect results
    language_results = []
    for batch in tqdm(batches, desc="Processing language batches"):
        language_results.extend(detect_language_batch(batch))
    
    # Assign detected languages back to the DataFrame
    df['language'] = language_results
    return df

# Threaded detection with langid
def detect_language_langid_threaded(df, max_workers=16):
    # Ensure 'truncated_text' contains only strings
    df['truncated_text'] = df['truncated_text'].astype(str).fillna('')

    # Define detection function for threading
    def detect_language_thread(text):
        return classify(text)[0] if text.strip() else "unknown"
    
    # Use ThreadPoolExecutor for concurrent detection
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        languages = list(tqdm(executor.map(detect_language_thread, df['truncated_text']), total=len(df), desc="Threaded language detection"))
    
    # Assign detected languages back to the DataFrame
    df['language'] = languages
    return df

# Main execution (assuming df and meta_df are passed in the notebook)
if __name__ == '__main__':
    print("Starting full pipeline...")

    # Run preprocessing
    df = run_processing(df, meta_df)

    # Run language detection
    original_row_count = len(df)  # Store the original row count before filtering

    # Use either batch or threaded method for langid
    df = detect_language_langid_threaded(df, max_workers=16)  # Threaded version
    # Alternatively, use batch processing:
    # df = detect_language_langid_batch(df, batch_size=1000)

    # Filter out non-English reviews
    df = df[df['language'] == 'en']

    # Drop unnecessary columns
    df = df[['rating', 'title', 'text', 'parent_asin', 
             'review_timestamp', 'helpful_vote', 'cleaned_text']]

    # Display the number of rows before and after filtering
    print(f"Final number of rows after removing non-English reviews: {len(df)}")

    # Calculate and print the percentage of rows filtered out
    filtered_out_percentage = ((original_row_count - len(df)) / original_row_count) * 100
    print(f"Percentage removed in language filter: {filtered_out_percentage}")
    
# Rename the 'title' column in df to 'review_text'
df.rename(columns={'title': 'review_title'}, inplace=True)

# Extract the relevant columns from the metadata dataset
meta_df_key = meta_df[['parent_asin', 'title', 'main_category', 'details']].copy()

# Function to inspect the 'details' field more thoroughly
def inspect_details(details):
    if isinstance(details, str):
        try:
            details = ast.literal_eval(details)  # Convert string representation of dictionary to actual dictionary
        except:
            return None
    return details

# Apply the inspection function
meta_df_key['details'] = meta_df_key['details'].apply(inspect_details)

# Enhanced function to search through nested dictionaries and lists for keys (case-insensitive)
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

# Merge the datasets together to get the titles and everything with progress
merged_df = pd.merge(df, meta_df_key, on='parent_asin', how='left')

# Calculate the percentage and count of parent_asins that successfully merged
total_parent_asins = df['parent_asin'].nunique()
merged_parent_asins = merged_df['parent_asin'].nunique()
merged_percentage = (merged_parent_asins / total_parent_asins) * 100

print(f"Percentage of parent_asins that merged: {merged_percentage:.2f}%")

# Calculate the percentage and count of records with 'item_model_number'
total_records = len(merged_df)
records_with_model_number = merged_df['item_model_number'].notna().sum()
model_number_percentage = (records_with_model_number / total_records) * 100

# Define a list of known invalid terms
unknown_terms = ['unknown', 'none', 'n/a', 'na', 'nan']

# Define a refined function to filter out invalid terms (but retain NaN values)
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

# Apply the refined filter to the 'item_model_number' column in merged_df
filtered_df = merged_df[~merged_df['item_model_number'].apply(is_invalid_item_model_number)]

# Calculate the summary statistics
filtered_records = len(filtered_df)
filtered_out_count = len(merged_df) - filtered_records
filtered_percentage = (filtered_out_count / len(merged_df)) * 100

print(f"\nFiltered out records with invalid 'item_model_number': {filtered_out_count}")

# Assign the filtered result back to merged_df
merged_df = filtered_df
# Show the first 5 records after filtering

def save_to_csv_with_encoding_handling(df, filename):
    try:
        df.to_csv(filename, index=False, encoding="utf-8", errors="replace")  # Replaces any invalid characters
    except UnicodeEncodeError as e:
        print(f"UnicodeEncodeError: {e}. Attempting to save with a different encoding.")
        df.to_csv(filename, index=False, encoding="utf-8", errors="replace")
        print(f"File saved as {filename} with encoding errors replaced.")

# Use the function to save your DataFrame
save_to_csv_with_encoding_handling(merged_df, "checkpoint1.csv")

def remove_invalid_unicode(s):
    return s.encode('utf-8', 'replace').decode('utf-8')

merged_df = merged_df.applymap(lambda x: remove_invalid_unicode(str(x)))

invalid_model_numbers = merged_df[merged_df['item_model_number'] == ''].shape[0]
# Proceed with your previous analysis
# Count the number of records with and without 'item_model_number'
model_number_counts = merged_df['item_model_number'].notna().value_counts()

# Ensure both True and False are present in the count (fill missing with 0)
model_number_counts = model_number_counts.reindex([True, False], fill_value=0)