import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# === Step 1: Load the raw dataset ===
df = pd.read_csv("data/spam_data.csv", names=["label", "spam_text"], encoding="latin1")

# Drop the header row that was read as data
df = df[df['label'].str.lower() != 'category']

# Drop duplicate headers and missing text
df = df[df['label'].str.lower() != 'label']
df.dropna(subset=['spam_text'], inplace=True)

# Reset index after filtering
df.reset_index(drop=True, inplace=True)

# === Step 2: Clean the text ===
def clean_text(text):
    text = str(text).lower()  # Convert to string and lowercase
    # Remove special characters (keep only ASCII letters and spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Keep only lowercase letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # normalize whitespace
    return text

df['spam_text_clean'] = df['spam_text'].apply(clean_text)

# === Step 3: Tokenization ===
def tokenize(text):
    tokens = re.split(r'\W+', text)
    return [t for t in tokens if t]  # Remove empty strings

df['spam_text_tokenized'] = df['spam_text_clean'].apply(tokenize)

# === Step 4: Remove stopwords ===
stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words and word != '']

df['spam_text_nonstop'] = df['spam_text_tokenized'].apply(remove_stopwords)

# === Step 5: Stemming ===
stemmer = PorterStemmer()
df['spam_text_stemmed'] = df['spam_text_nonstop'].apply(lambda x: [stemmer.stem(word) for word in x])

# === Step 6: Lemmatization ===
lemmatizer = WordNetLemmatizer()
df['spam_text_lemmatized'] = df['spam_text_nonstop'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# === Step 7: Save final file ===
df.to_csv("data_processing/spam_data_cleaned.csv", index=False)