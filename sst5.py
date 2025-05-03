import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import joblib

# Define project path
base_path = "/home/vdsat/Desktop/Projects/SENTIMENT_ANALYSIS-main"

# Loading datasets
def load_data(file_path):
    """Load dataset from tab-separated file (label\ttext)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    dataset = pd.read_csv(file_path, sep='\t', header=None, names=["label", "text"])
    dataset['label'] = dataset['label'] - 1  # Adjust labels from 1-5 to 0-4
    return dataset

train_dataset = load_data(os.path.join(base_path, "train_modified.txt"))
test_dataset = load_data(os.path.join(base_path, "test_modified.txt"))
dev_dataset = load_data(os.path.join(base_path, "dev_modified.txt"))

# Apply custom mapping
def remap_label(label):
    if label in [0, 1]:
        return 0  # Negative
    elif label == 2:
        return 2  # Neutral
    else:  # label in [3, 4]
        return 4  # Positive

train_dataset['label'] = train_dataset['label'].apply(remap_label)
test_dataset['label'] = test_dataset['label'].apply(remap_label)
dev_dataset['label'] = dev_dataset['label'].apply(remap_label)

train_sentences = train_dataset['text']
train_labels = train_dataset['label']
test_sentences = test_dataset['text']
test_labels = test_dataset['label']
dev_sentences = dev_dataset['text']
dev_labels = dev_dataset['label']

# Concatenate train and test for vectorization
sentences = pd.concat([train_sentences, dev_sentences], ignore_index=True)
labels = pd.concat([train_labels, dev_labels], ignore_index=True)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(
    stop_words='english',
    lowercase=True,
    ngram_range=(1, 2),
    max_features=10000
)
tfidf_matrix = vectorizer.fit_transform(sentences)
test_tfidf_matrix = vectorizer.transform(test_sentences)

# Train logistic regression classifier
classifier = LogisticRegression(
    C=10,
    penalty='l2',
    solver='lbfgs',
    max_iter=10000,
    class_weight='balanced'
)
classifier.fit(tfidf_matrix, labels)

# Evaluate on dev set
y_pred = classifier.predict(test_tfidf_matrix)
accuracy = accuracy_score(test_labels, y_pred)
print(f"Dev Accuracy: {accuracy:.4f}")
print("Classification Report (Dev Set):")
print(classification_report(test_labels, y_pred, target_names=["Negative", "Neutral", "Positive"]))

# Save model and vectorizer
model_path = os.path.join(base_path, "sst5_logistic_model.joblib")
vectorizer_path = os.path.join(base_path, "tfidf_vectorizer.joblib")
joblib.dump(classifier, model_path)
joblib.dump(vectorizer, vectorizer_path)
print(f"Model saved to '{model_path}'")
print(f"Vectorizer saved to '{vectorizer_path}'")

# Negation handling for predictions
negation_words = {
    "not", "no", "never", "none", "nobody", "nothing", "neither",
    "nowhere", "hardly", "barely", "scarcely", "isn't", "wasn't",
    "weren't", "doesn't", "don't", "didn't", "won't", "wouldn't",
    "can't", "couldn't", "shouldn't", "cannot"
}

def contains_negation(sentence):
    words = sentence.lower().split()
    return any(word in negation_words for word in words)

def predict_sentiment(sentence):
    sentence = sentence.lower().strip()
    test = vectorizer.transform([sentence])
    pred = classifier.predict(test)[0]
    if not contains_negation(sentence):
        if pred == 0:
            out = "Negative"
        elif pred == 2:
            out = "Neutral"
        else:
            out = "Positive"
    else:
        if pred == 0:
            out = "Neutral"
        elif pred == 2:
            out = "Negative"
        else:
            out = "Neutral"
    return out