# import spacy
# import benepar
# from nltk import Tree
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np
import pandas as pd

# Loading datasets
train_dataset = pd.read_csv('train_modified.txt', sep='\t', header=None, names=["label", "text"])
train_dataset['label'] = train_dataset['label'] - 1

test_dataset = pd.read_csv('test_modified.txt', sep='\t', header=None, names=["label", "text"])
test_dataset['label'] = test_dataset['label'] - 1 

dev_dataset = pd.read_csv('dev_modified.txt', sep='\t', header=None, names=["label", "text"])
dev_dataset['label'] = dev_dataset['label'] - 1

# Apply custom mapping
def remap_label(label):
    if label in [0, 1]:
        return 0
    elif label == 2:
        return 2
    else:  # label in [3, 4]
        return 4

train_dataset['label'] = train_dataset['label'].apply(remap_label)
test_dataset['label'] = test_dataset['label'].apply(remap_label)
dev_dataset['label'] = dev_dataset['label'].apply(remap_label)

train_sentences = train_dataset['text']
train_labels = train_dataset['label']
test_sentences = test_dataset['text']
test_labels = test_dataset['label']
dev_sentences = dev_dataset['text']
dev_labels = dev_dataset['label']

sentences = pd.concat([train_sentences, test_sentences], ignore_index=True)
labels = pd.concat([train_labels, test_labels], ignore_index=True)

# sentences = pd.concat([sentences, dev_sentences], ignore_index=True)
# labels = pd.concat([labels, dev_labels], ignore_index=True)

# Convert text sentences to number form using tfidf vectorizer
vectorizer = TfidfVectorizer(
    stop_words='english',  # Use scikit-learn's English stop words
    lowercase=True,
    ngram_range=(1, 2),  # add bigrams
    max_features=10000
)

tfidf_matrix = vectorizer.fit_transform(sentences)
dev_tfidf_matrix = vectorizer.transform(dev_sentences)
# Define classifier and fit training data
classifier = LogisticRegression(C=10, penalty='l2', solver='lbfgs', max_iter=10000, class_weight='balanced')
classifier.fit(tfidf_matrix, labels)

y_pred = classifier.predict(dev_tfidf_matrix)
accuracy = accuracy_score(dev_labels, y_pred)
print(f"Training accuracy: {accuracy}")

def print_accuracy():
    return accuracy

# Load Spacy model and Benepar once at the start
# nlp = spacy.load("en_core_web_md")
# benepar.download('benepar_en3')
# nlp.add_pipe("benepar", config={"model": "benepar_en3"})

# Function to extract meaningful phrases from a sentence
# def phrases_from_sentence(sentence):
#     doc = nlp(sentence)  # Process the sentence with spacy and benepar
#     phrases = []
#     for sent in doc.sents:
#         tree = sent._.parse_string  # Get the parse tree for the sentence
#         parsed = Tree.fromstring(tree)  # Parse the tree using NLTK
#         # Extract all phrases where the subtree height is greater than 2
#         phrases += [' '.join(leaf) for subtree in parsed.subtrees() if subtree.height() > 2 for leaf in [subtree.leaves()]]
#     return phrases

# Function to predict sentiment for a phrase
# def predict_sentiment_phrase(phrase):
#     test = vectorizer.transform([phrase])
#     pred = classifier.predict(test)
#     return pred[0]

# Function to predict sentiment for the sentence by analyzing its phrases
# def predict_sentiment_sentence(sentence):
#     phrases = phrases_from_sentence(sentence)
#     len_phrases = len(phrases)
#     value = 0
#     out = {}
#     for i in reversed(range(len_phrases)):
#         p = predict_sentiment_phrase(phrases[i])
#         out[phrases[i]] = p
#         if p != 2:
#             value += p
#     value /= len_phrases  # Compute average sentiment score based on phrases

#     if value < 1.5:
#         out = "This review is negative"
#     elif value < 2.5:
#         out = "This review is neutral"
#     else:
#         out = "This review is positive"
    
#     return out

def predict_sentiment(sentence):
    sentence = sentence.lower().strip()
    test = vectorizer.transform([sentence])
    pred = classifier.predict(test)
    if pred[0] == 0 or pred[0] == 1:
        out = "This is negative review"
    elif pred[0] == 2:
        out = "This review was neutral"
    else:
        out = "This review was positive"
    return out
