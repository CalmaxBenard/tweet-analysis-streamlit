import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

df = pd.read_csv("airline_tweets_data.csv")

st.set_page_config(page_title="sentiments analyzer", page_icon=":pager:", layout="wide")
st.title("Tweets Analysis")
st.header("Quick Overview")
st.write(df.head())

col1, col2 = st.columns(2)

with (col1):
    st.write("Compare the Sentiments")
    fig = plt.figure(dpi=200)
    sns.countplot(data=df, x="airline_sentiment")
    st.pyplot(fig)

with (col2):
    st.write("Reasons for the Negative Ratings")
    fig = plt.figure(dpi=200)
    sns.countplot(data=df, x="negativereason", hue="negativereason")
    plt.xticks(rotation=90)
    st.pyplot(fig)

st.write("Comparison of Sentiments across the Airlines")
fig = plt.figure()
sns.countplot(data=df, x="airline", hue="airline_sentiment")
st.pyplot(fig)

st.header("ML Model")

X = df["text"]
y = df["airline_sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

tfidf = TfidfVectorizer(stop_words="english")
tfidf.fit(X_train)

X_train_tfidf = tfidf.transform(X_train)
X_test_tfdif = tfidf.transform(X_test)

# Naive_Bayes Model
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

# LogisticRegression Model
log_model = LogisticRegression(max_iter=100000)
log_model.fit(X_train_tfidf, y_train)

# SVM Model
svc = SVC()
svc.fit(X_train_tfidf, y_train)

# linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train_tfidf, y_train)


def report(model):
    preds = model.predict(X_test_tfdif)
    st.write(classification_report(y_test, preds))
    st.write(confusion_matrix(y_test, preds))


# st.header("Naive Bayes Model")
# report(nb)
# st.header("Logistic Regression")
# report(log_model)
# st.header("Support Vector Machine Model")
# report(svc)
# st.header("Linear SVC")
# report(linear_svc)

pipe = Pipeline([("tfidf", TfidfVectorizer()), ("svc", LinearSVC())])
pipe.fit(X, y)

st.header("Customer Reviews")
text = st.text_area("Enter your Review")
prediction = pipe.predict([text])

if prediction is not None:
    st.header(f"The sentiment is {prediction[0]}")



