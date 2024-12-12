final_columns = [
    'rating', 'review_title', 'text', 'parent_asin', 'review_timestamp',
    'cleaned_text', 'title', 'item_model_number', 'manufacturer', 'brand',
    'UPC', 'involved_in_incident', 'incident_report_date', 'incident_description', 
    'incident_purchase_date', 'Manufacturer / Importer / Private Labeler Name',
    'Brand']
csv_file_path = 'checkpoint2.csv'

try:
    print("Reloading for sentiment analysis...")
    chunk_size = 1000000
    df_list = []
    for chunk in pd.read_csv(
        csv_file_path, usecols=final_columns,
        chunksize=chunk_size, engine='python'
    ):
        df_list.append(chunk)
    df = pd.concat(df_list, ignore_index=True)
except pd.errors.ParserError as e:
    print(f"Parser error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

# Load model and tokenizer on GPU if available
nltk.download("vader_lexicon")
sid = SentimentIntensityAnalyzer()

# Function to truncate text based on character length
def truncate_text_batch(df_chunk, max_length=2016):
    df_chunk['truncated_text'] = df_chunk['text'].str.slice(0, max_length)
    
    # Count truncated rows
    truncation_flags = df_chunk['text'].str.len() > max_length
    affected_count = truncation_flags.sum()
    
    return df_chunk, affected_count

# Main function to handle truncation in parallel batches
def apply_truncate_text_in_batches(df, max_length=2016, batch_size=10000):
    chunks = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
    
    affected_total = 0
    results = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(truncate_text_batch, chunk, max_length) for chunk in chunks]
        for future in tqdm(futures, desc="Truncating Text in Parallel", leave=True, position=0):
            df_chunk, affected_count = future.result()
            results.append(df_chunk)
            affected_total += affected_count
    
    truncated_df = pd.concat(results, ignore_index=True)
    total_count = len(df)
    percentage_affected = (affected_total / total_count) * 100
    
    return truncated_df

# VADER sentiment scoring function
def vader_sentiment_analysis(text):
    if not isinstance(text, str):  # Ensure text is a string
        text = ""
    scores = sid.polarity_scores(text)
    return scores['compound']

# Function to classify sentiment based on rating
def classify_based_on_rating(rating):
    if rating in [1, 2]:
        return 'NEGATIVE'
    elif rating == 3:
        return 'NEUTRAL'
    elif rating in [4, 5]:
        return 'POSITIVE'

# Function to apply sentiment analysis in batches
def vader_sentiment_analysis_batch(df_chunk):
    # Pre-classify based on rating
    df_chunk['sentiment_score_preclass'] = df_chunk['rating'].apply(classify_based_on_rating)

    # Run VADER sentiment analysis
    sentiment_scores = []
    for text in df_chunk['truncated_text']:
        sentiment_scores.append(vader_sentiment_analysis(text))
    df_chunk['sentiment_score_raw'] = sentiment_scores

    return df_chunk

# Main function to process sentiment analysis in loops and batches
def process_sentiment_analysis(df, batch_size=10000):
    chunks = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
    results = []

    # Single progress bar for the entire process
    with tqdm(total=len(df), desc="Processing Sentiment Analysis", leave=True, position=0) as pbar:
        for chunk in chunks:
            processed_chunk = vader_sentiment_analysis_batch(chunk)
            results.append(processed_chunk)
            pbar.update(len(chunk))  # Update the progress bar by the chunk size

    return pd.concat(results, ignore_index=True)

# Apply truncation
df = apply_truncate_text_in_batches(df, max_length=2016, batch_size=10000)

# Apply sentiment analysis
df = process_sentiment_analysis(df, batch_size=10000)

# Adjust the sentiment score based on rating
df['sentiment_score'] = df['sentiment_score_raw']

# VADER sentiment scoring function, skipping empty descriptions
def vader_sentiment_analysis(text):
    if isinstance(text, str) and text.strip():
        score = sid.polarity_scores(text)['compound']
        label = "POSITIVE" if score > 0.05 else "NEGATIVE" if score < -0.05 else "NEUTRAL"
        return score, label
    return None, None

# Run VADER sentiment analysis on `incident_description` with both score and label
tqdm.pandas(desc="Running VADER Sentiment Analysis on Incident Description")
df[['incident_desc_sentiment_score', 'incident_desc_sentiment_label']] = df['incident_description'].progress_apply(
    lambda x: pd.Series(vader_sentiment_analysis(x))
)

# Function to remove invalid characters in strings
def clean_text(text):
    return text.encode('utf-8', 'replace').decode('utf-8')

# Apply the cleaning function to all string columns
df = df.applymap(lambda x: clean_text(x) if isinstance(x, str) else x)

# Now save to CSV
df.to_csv("checkpoint3.csv", index=False, encoding='utf-8')