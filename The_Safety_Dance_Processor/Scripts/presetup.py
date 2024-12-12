import nltk
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import subprocess

# Install required packages
required_packages = [
    "nltk", "spacy", "transformers", "faiss-gpu", "sentence-transformers"
]

print("Installing required packages...")
subprocess.run(["pip", "install"] + required_packages)

# NLTK downloads
nltk_resources = [
    "stopwords", "wordnet", "omw-1.4", "words", "vader_lexicon"
]

print("Downloading NLTK resources...")
for resource in nltk_resources:
    nltk.download(resource)

# Spacy model downloads
spacy_models = ["en_core_web_sm", "en_core_web_lg"]

print("Downloading Spacy models...")
for model in spacy_models:
    try:
        spacy.load(model)
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", model])

# Hugging Face model downloads
huggingface_models = [
    "google/bigbird-roberta-base",
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
]

print("Downloading Hugging Face models...")
for model_name in huggingface_models:
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForSequenceClassification.from_pretrained(model_name)

print("Setup complete!")