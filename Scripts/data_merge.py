# Data Loading and Preprocessing
script_dir = os.path.dirname(os.path.abspath(__file__))

def load_category():
    try:
        with open("chosen_category.txt", "r") as f:
            category = f.read().strip()  # Strip any extra whitespace
        return category
    except FileNotFoundError:
        print("Error: 'chosen_category.txt' not found. Ensure the category is selected first.")
        return None
category = load_category()

# Build the full path to the file
csv_file_path = os.path.join(script_dir, "Incidents_Extracted_model_number_cleaned2.csv")

# Data Loading and Preprocessing
try:
    incident_df = pd.read_csv(csv_file_path, encoding='ISO-8859-1', on_bad_lines='skip')
except FileNotFoundError:
    print(f"File not found: {csv_file_path}")
    raise

for col in merged_df.columns:
    if merged_df[col].dtype == 'int64':
        merged_df[col] = merged_df[col].astype('int32')
    elif merged_df[col].dtype == 'float64':
        merged_df[col] = merged_df[col].astype('float32')

def remove_spaces_and_dashes(text):
    if isinstance(text, str):
        return text.replace(' ', '').replace('-', '').replace('.', '')
    return text


# Ensure all relevant columns in merged_df are in lowercase, and removed spaces and dashes
merged_df['parent_asin'] = merged_df['parent_asin'].astype(str).str.lower()
merged_df['item_model_number'] = merged_df['item_model_number'].astype(str).str.lower()
merged_df['item_model_number'] = merged_df['item_model_number'].apply(remove_spaces_and_dashes)
merged_df['UPC'] = merged_df['UPC'].apply(remove_spaces_and_dashes)

incident_df.columns = incident_df.columns.str.strip()
columns_to_select = [
    'Report Date', 'Product Description', 'Manufacturer / Importer / Private Labeler Name',
    'Brand', 'Extracted Model Number', 'Purchase Date', 'Retailer', 
    'Incident Description'
]
try:
    incident_df = incident_df[columns_to_select]
except KeyError as e:
    missing_columns = [col for col in columns_to_select if col not in incident_df.columns]
    print(f"Missing columns: {missing_columns}")
    raise

# convert incident_df columns to lowercase, remove spaces and dashes
incident_df['Extracted Model Number'] = incident_df['Extracted Model Number'].apply(remove_spaces_and_dashes)
incident_df = incident_df.apply(lambda col: col.str.lower() if col.dtype == 'object' else col)
# Output a CSV for QC(test2)
#incident_df.to_csv('test2.csv', index=False)

# cleanup time
gc.collect()

english_words = set(words.words())

def clean_text(text):
    # Remove any characters that are not valid UTF-8
    if isinstance(text, str):
        return text.encode('utf-8', 'ignore').decode('utf-8')
    return text

# Clean text columns in the DataFrames
merged_df = merged_df.applymap(clean_text)
incident_df = incident_df.applymap(clean_text)

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

# Output a CSV for QC after exploding ASINs, model numbers, and UPCs (test4)
#incident_df_exploded.to_csv('test4.csv', index=False)

# Build UPC regex pattern for fast substring matching in chunks
unique_upcs = incident_df_exploded['extracted_upcs'].dropna().unique()
upc_pattern = "|".join(re.escape(upc) for upc in unique_upcs)

# Check if the CSV file exists and delete it to start fresh
csv_file_path = 'checkpoint2.csv'
if os.path.exists(csv_file_path):
    os.remove(csv_file_path)
    print("Existing intermediate CSV file removed. Starting fresh...")

print(f"\nMemory cleanup and optimization completed. Proceeding with the merging process...")


# Define the chunk size (adjust based on your available memory)
chunk_size = 1000000
total_records = len(merged_df)

# Variables to track statistics
total_asin_matches = 0
total_model_number_matches = 0
total_upc_matches = 0
total_involved_rows = 0

upc_pattern_dict = {
    row['extracted_upcs']: f".*{re.escape(row['extracted_upcs'])}.*" 
    for _, row in incident_df_exploded.dropna(subset=['extracted_upcs']).iterrows()
}

# Dictionary to map UPCs to their incident data for lookup
incident_info_dict = {
    upc: (row['Report Date'], row['Incident Description'], row['Purchase Date'])
    for upc, row in incident_df_exploded.dropna(subset=['extracted_upcs']).set_index('extracted_upcs').iterrows()
}

# Function to match ASINs, model numbers, and UPCs using exploded incident data
def match_asins_model_numbers_and_upc(chunk, incident_df_exploded):
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
    print(f"ASIN matches found: {chunk['asin_match'].sum()} in this chunk")

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
    print(f"Short model number matches found: {chunk['model_number_match'].sum()} in this chunk")
    
    # For long model numbers (greater than 6 characters)
    chunk = pd.merge(chunk, long_model_numbers[['extracted_model_numbers', 'Report Date', 'Incident Description', 'Purchase Date']],
                     how='left', left_on='item_model_number', right_on='extracted_model_numbers', suffixes=('', '_model'))
    chunk.loc[chunk['extracted_model_numbers_model'].notna(), 'model_number_match'] = True
    print(f"Long model number matches found: {chunk['model_number_match'].sum()} in this chunk")

    # UPC and Model ID matching using Aho-Corasick automaton
    def find_upc_matches(text):
        if not isinstance(text, str):
            return []
        return [upc for _, upc in A.iter(text)]

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
    print(f"UPC matches found: {chunk['upc_match'].sum()} in this chunk")

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
    print(f"Total reviews involved in an incident in this chunk: {chunk['involved_in_incident'].sum()} (after deduplication).")
    return chunk

# Process the merged_df in chunks
for start in range(0, total_records, chunk_size):
    end = min(start + chunk_size, total_records)
    chunk = merged_df.iloc[start:end].copy()

    print(f"Processing records from {start} to {end}...")

    # Perform the optimized ASIN, model number, and UPC matching
    chunk = match_asins_model_numbers_and_upc(chunk, incident_df_exploded)

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
print("\n--- Merge Complete. Incident Dataset Merge: Summary Statistics ---")
print(f"Percentage of records with incidents: {total_percentage:.2f}%")
print(f"Total ASIN matches: {total_asin_matches}")
print(f"Total model number matches: {total_model_number_matches}")
print(f"Total UPC matches: {total_upc_matches}")

final_columns = [
    'rating', 'review_title', 'text', 'parent_asin', 'review_timestamp',
    'cleaned_text', 'title', 'item_model_number', 'manufacturer', 'brand',
    'UPC', 'involved_in_incident', 'incident_report_date', 'incident_description', 
    'incident_purchase_date', 'Manufacturer / Importer / Private Labeler Name',
    'Brand']
df = pd.read_csv(csv_file_path, usecols=final_columns)

unique_involved_products = df[df['involved_in_incident'] == True]['parent_asin'].nunique()

print(f"\nTotal unique products involved in incidents: {unique_involved_products}")


# Filter for rows where involved_in_incident is True
df_involved = pd.read_csv(csv_file_path)
df_involved = df_involved[df_involved['involved_in_incident'] == True]

# Save this filtered dataframe to a separate CSV
df_involved.to_csv(f"incident_involved_records_{category}.csv", index=False)
