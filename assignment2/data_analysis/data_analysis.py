# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('wordnet')

# Load the cleaned dataset
df = pd.read_csv("../data_processing/spam_data_cleaned.csv")
print(df.head())

# Check label distribution
values = df['label'].value_counts()
total = values.sum()

percentage_0 = (values['ham'] / total) * 100
percentage_1 = (values['spam'] / total) * 100

print('Percentage of ham:', percentage_0)
print('Percentage of spam:', percentage_1)

# Count plot of labels
cols = ["#04B2D9", "#53fca1"]
plt.figure(figsize=(10, 5))
fg = sns.countplot(x=df['label'], legend=False, palette=cols)
fg.set_title("Count Plot of Classes")
fg.set_xlabel("Classes")
fg.set_ylabel("Number of Data Points")
plt.show()

# Pairplot
df["No_of_Characters"] = df["spam_text"].apply(len)
df["No_of_Words"] = df["spam_text"].apply(lambda x: len(nltk.word_tokenize(x)))
df["No_of_Sentences"] = df["spam_text"].apply(lambda x: len(nltk.sent_tokenize(x)))
df.describe().T
sns.pairplot(data=df, hue="label", palette=cols)
plt.show()

# Word Cloud
stop_word = set(stopwords.words('english'))
word_cloud = WordCloud(width=800, height=800, max_words=200, stopwords=stop_word, background_color='black', max_font_size=200)

spam = df.query("label == 'spam'")['spam_text'].str.cat(sep=' ')
ham = df.query("label == 'ham'")['spam_text'].str.cat(sep=' ')

print("Spam Word Cloud")
word_cloud.generate(spam)
plt.figure(figsize=(16, 8))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()

print("Ham Word Cloud")
word_cloud.generate(ham)
plt.figure(figsize=(16, 8))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Encode label for correlation
if "No_of_Characters" not in df.columns:
    df["No_of_Characters"] = df["spam_text"].apply(len)

if "No_of_Words" not in df.columns:
    df["No_of_Words"] = df["spam_text"].apply(lambda x: len(nltk.word_tokenize(x)))

if "No_of_Sentences" not in df.columns:
    df["No_of_Sentences"] = df["spam_text"].apply(lambda x: len(nltk.sent_tokenize(x)))

if "label_encoded" not in df.columns:
    df["label_encoded"] = df["label"].map({"ham": 0, "spam": 1})
    
# Correlation matrix
corr_features = df[["No_of_Characters", "No_of_Words", "No_of_Sentences", "label_encoded"]]
corr_matrix = corr_features.corr()

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Between Features and Label")
plt.show()