import joblib
import gradio as gr
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ensure stopwords are available
nltk.download("stopwords")

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# load model + vectorizer
model = joblib.load("spam_model.joblib")
cv = joblib.load("vectorizer.joblib")

def clean_text(text: str) -> str:
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower()
    words = review.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

def predict_spam(message: str) -> str:
    cleaned = clean_text(message)
    vec = cv.transform([cleaned]).toarray()
    pred = model.predict(vec)[0]
    return "Spam" if pred == 1 else "Ham"

demo = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(lines=3, label="Enter SMS message"),
    outputs=gr.Label(label="Prediction"),
    title="SMS Spam Classifier",
    description="Classifies SMS messages as Spam or Ham.",
)

if __name__ == "__main__":
    demo.launch()

