import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading data...")
df = pd.read_csv("data_processing/spam_data_cleaned.csv")

X = df['spam_text']
y = df['label'].map({'spam': 1, 'ham': 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(
    max_features=3000, lowercase=True, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

with open('machine_learning/models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Training models...")

models = {}
results = {}

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_vec, y_train)
models['Logistic Regression'] = lr

nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
models['Naive Bayes'] = nb

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_vec, y_train)
models['Random Forest'] = rf

svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train_vec, y_train)
models['SVM'] = svm

print("Evaluating...")

for name, model in models.items():
    preds = model.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results[name] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'predictions': preds
    }

    print(f"{name}: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")

for name, model in models.items():
    filename = name.replace(' ', '_').lower()
    with open(f'machine_learning/models/{filename}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

with open('machine_learning/models/model_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# comparison chart
names = list(results.keys())
accs = [results[m]['accuracy'] for m in names]
precs = [results[m]['precision'] for m in names]
recs = [results[m]['recall'] for m in names]
f1s = [results[m]['f1_score'] for m in names]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(names))
width = 0.2

ax.bar(x - 1.5*width, accs, width, label='Accuracy')
ax.bar(x - 0.5*width, precs, width, label='Precision')
ax.bar(x + 0.5*width, recs, width, label='Recall')
ax.bar(x + 1.5*width, f1s, width, label='F1-Score')

ax.set_xlabel('Models')
ax.set_ylabel('Score')
ax.set_title('Model Comparison')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend()

plt.tight_layout()
plt.savefig('report/model_comparison.png', dpi=150)
plt.close()

# confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

for idx, (name, model) in enumerate(models.items()):
    preds = results[name]['predictions']
    cm = confusion_matrix(y_test, preds)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    axes[idx].set_title(f'{name}')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('report/confusion_matrices.png', dpi=150)
plt.close()

print("Done!")
