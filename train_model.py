from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

texts = [
    "This is written by human",
    "AI generated text example"
]

labels = [0, 1]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)

pickle.dump((model, vectorizer), open("model/model.pkl", "wb"))