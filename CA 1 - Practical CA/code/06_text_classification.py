# ============ TEXT CLASSIFICATION (e.g. spam / sentiment) ============

# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# %% 1. Load data
df = pd.read_csv("../../experiments/spam_classifier/SMSSpamCollection.txt",
                 sep="\t", names=["label", "text"])        # <-- change
print(df["label"].value_counts())

# %% 2. Simple cleaning: lowercase + remove punctuation
df["text"] = df["text"].str.lower()
df["text"] = df["text"].str.replace(r"[^a-z0-9 ]", " ", regex=True)

# %% 3. Split (split BEFORE vectorising)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

# %% 4. Option A: Bag of Words (word counts) + Naive Bayes
bow = CountVectorizer(stop_words="english")
X_train_bow = bow.fit_transform(X_train)     # learn vocabulary on train only
X_test_bow = bow.transform(X_test)
nb = MultinomialNB()
nb.fit(X_train_bow, y_train)
print("BoW + NB accuracy:", accuracy_score(y_test, nb.predict(X_test_bow)))

# %% 5. Option B: TF-IDF (down-weights common words) + Logistic Regression
tfidf = TfidfVectorizer(stop_words="english")
X_train_tf = tfidf.fit_transform(X_train)
X_test_tf = tfidf.transform(X_test)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tf, y_train)
pred = lr.predict(X_test_tf)
print("TF-IDF + LR accuracy:", accuracy_score(y_test, pred))

# %% 6. Evaluate
print(classification_report(y_test, pred))
ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.show()

# %% 7. Top words that push towards the positive class (explainability)
words = pd.Series(lr.coef_[0], index=tfidf.get_feature_names_out())
print(words.sort_values().tail(15))
