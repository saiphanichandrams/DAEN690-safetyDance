**############################THE SAFETY DANCE##############################**</br>
################################## Amazon Review Safety Analysis Pipeline ####################################
######################### By Jonathan King, Aakash Boenal,Utkarsh Ganjihal, SaiPhani Chandra Vuppala, DAEN690 - Fall 2024 - George Mason University ####################
This program is a large-scale Amazon review analysis tool that downloads, preprocesses, and analyzes review 
data to identify potentially unsafe products. The tool employs both heuristic analysis and machine learning 
(Isolation Forest) to flag products based on various features, such as ratings, sentiment, and safety-related 
terms.

Developed as a Data Analytics Capstone Project (DAEN690 - Fall 2024) by Jonathan King, Aakash Boenal, Utkarsh Ganjihal, SaiPhani Chandra Vuppala, George Mason University, 
this program is optimized for large datasets and advanced analysis.

Partnered with NIRA, Inc for real-world analysis.

############################################# FEATURES #######################################################
Category-based Review Download: 
- Downloads Amazon review data by category from the McAuley Lab dataset.
Preprocessing: 
- Cleans, normalizes, and filters non-English reviews using parallel processing.
Sentiment Analysis: 
- Applies VADER to classify review sentiment (positive, neutral, negative).
Safety Term Extraction: 
- Matches predefined safety-related terms and calculates overlap with incident descriptions.
Heuristic Analysis: 
- Flags potentially unsafe products based on quantile thresholds:
- 25th percentile: Low ratings and sentiment scores.
- 75th percentile: High safety term density and incident-review match rates.
Isolation Forest: 
- Uses machine learning to detect outliers based on safety-related features.

############################################# SETUP ##########################################################
Required Packages
- Installation handled in the first cell of the main notebook 
- Or via requirements.txt (for Conda/virtual environments).
Core dependencies include:
- scikit-learn, pandas, numpy, tqdm, spacy, nltk, transformers, datasets
Data Sources
McAuley Lab Amazon Reviews Dataset:
- Installed via pip install datasets from Hugging Face.
Incidents Data:
- Static file: Incidents_Extracted_model_number_cleaned2.csv.
- Includes manually extracted ASINs, Model IDs, UPCs, and TCINs.
Recalls Data:
- Static file: recall_cleaned_manual.csv.
- Includes attributes manually extracted from recall reports.

** NOTE**: Place all static files in the Scripts directory.

###################################### PIPELINE OVERVIEW #####################################################
- Main Notebook: Main_Notebook.ipynb
- This notebook runs the entire pipeline by invoking the following scripts in sequence:

**package_handling.py**
- Prepares the environment by importing required libraries, configuring CUDA for GPU acceleration, and 
ensuring NLP tools and models (e.g., BigBird, spaCy) are downloaded and set up for processing. It also handles 
multiprocessing setup and suppresses unnecessary warnings for a clean runtime.

**review_downloader.py**
- Provides the CLI for downloading Amazon reviews and metadata by category from the McAuley Lab dataset. It 
lists categories, facilitates user selection, converting them into DataFrames for all the later processing.

**filtering_preprocessing.py**
- Preprocesses review data by cleaning text (lowercasing, lemmatizing, and removing stopwords), 
converting timestamps, and truncating for language detection. Filters out non-English reviews using 
parallel/batch language detection. It then does the first merge, review and metadata datasets by extracting 
key product details (e.g., item model number, manufacturer, brand, UPC) while filtering to retain only valid 
model numbers, including alphanumeric checks and removal of ambiguous terms. Saves a checkpoint file.

**data_merge.py**
- This script merges Amazon review data with incident data, using ASINs, model numbers, and UPCs to identify 
matches. It processes datasets in chunks to handle memory constraints with consistent data formatting. Extracts 
ASINs, model numbers, and UPCs using regex and Aho-Corasick for efficient substring matching. Matches populate 
incident-related details like report date, description, and purchase date, with deduplication applied. Saves 
another checkpoint file.

**sentiment_analysis.py**
- The script performs sentiment analysis on on the merged dataset using the VADER SentimentIntensityAnalyzer, 
classifying reviews as positive, negative, or neutral, pre-classified based on the amazon review. Text is 
truncated, then sentiment scores are derived from both review text, with batch processing and parallelization.

**word_matching.py**
- This script identifies safety-related terms in Amazon reviews and incident descriptions by preprocessing and 
matching words against a predefined set of safety terms and their synonyms. It uses tools like WordNet to expand 
the safety terms with synonyms. It first cleans, normalizes, and tokenizes text data. Matches are performed on 
both reviews and incident descriptions to identify potential safety issues to later calculate match rates.

**recall_merge.py (optional)**
- This script performs a very similiar merge function as the data_merge.py script. This script has no bearing 
on the final data aggregation, is a relic from previous labelled data machine learning attempts and is only 
informational.

**aggregation_and_heuristic.py**
- This script aggregates review and incident data at the product level, calculating metrics for average 
ratings, sentiment scores, safety term densities, and incident-review match rates. It uses quantiles to set 
thresholds for identifying potentially unsafe products. Flagged are for "potentially unsafe" based on low 
ratings, low sentiment scores, high safety term densities, or high match rates between reviews and incident 
safety terms. Final results include a CSV of unsafe products (heuristic_unsafe_products_<category>.csv).

**isoforest.py**
- This is the final script, and uses Isolation Forest to detect anomalous products in the dataset based on 
six dimensions: average rating, review count, sentiment score, safety term density, and incident-review match 
rate. Products with fewer than 15 reviews are filtered out before training. Detected anomalies are flagged, 
counted, and saved in a CSV file (isolationforest_unsafe_products_<category>.csv) for further inspection. A 
pairwise feature comparison plot is saved to highlight the outliers.

**Big_Loop.py**
- Separated from the other scripts, this script compiles ALL categories and all functions. Be prepared to wait.


######################################## OUTPUT FILES #########################################################

Heuristic Analysis:
- heuristic_unsafe_products_<category>.csv: List of unsafe products based on heuristic thresholds.
Isolation Forest Analysis:
- isolationforest_unsafe_products_<category>.csv: Products flagged as anomalies.
- Pairwise plot saved as isolationforest_pairplot_products_<category>.jpeg.
