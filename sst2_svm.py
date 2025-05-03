from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.svm import SVC

data = load_dataset("sst2")

train_sentences = data['train']['sentence']

train_labels = data['train']['label']

test_sentences = data['test']['sentence']

test_labels = data['test']['label']

validation_sentences = data['validation']['sentence']

validation_labels = data['validation']['label']

vectorizer = TfidfVectorizer(
    stop_words='english',  # Use scikit-learn's English stop words
    lowercase=True,
    max_features=1000
)

train_tfidf_matrix = vectorizer.fit_transform(train_sentences)
validation_tfidf_matrix = vectorizer.transform(validation_sentences)
test_tfidf_matrix = vectorizer.transform(test_sentences)

model = SVC(kernel='linear')
model.fit(train_tfidf_matrix, train_labels)

y_pred = model.predict(validation_tfidf_matrix)
print(classification_report(validation_labels, y_pred))

test_pred = model.predict(test_tfidf_matrix)
