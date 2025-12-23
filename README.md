
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

1. README ka skeleton
Spam_classifier/README.md ko is structure mein likh (ya update kar):

text
# SMS Spam Classifier

A simple end‑to‑end SMS spam detector built with scikit‑learn and deployed on Hugging Face Spaces using Gradio.

- **Live demo:** https://huggingface.co/spaces/Aryanji123/Spam_classifier
- **Notebook:** `Spam_classifier.ipynb`
- **Model:** Multinomial Naive Bayes on bag‑of‑words features.

## 1. Project overview

This app classifies SMS messages as **spam** or **ham** using a classic NLP pipeline:

- Data: UCI SMS Spam Collection dataset (`SMSSpamCollection.txt`).
- Preprocessing: regex cleaning, lowercasing, stopword removal, Porter stemming.
- Features: `CountVectorizer` (bag of words, max_features=5000).
- Model: `MultinomialNB` from scikit‑learn.

## 2. Quick start (local)

git clone https://github.com/aryankd123/Spam_classifier.git
cd Spam_classifier
pip install -r requirements.txt
python app.py

text

Then open `http://127.0.0.1:7860` in the browser.

## 3. Model training

Training code is in `Spam_filter/Spam_classifier.ipynb`:

1. Load `SMSSpamCollection.txt` and create the `messages` DataFrame.
2. Clean and preprocess messages into a `corpus`.
3. Vectorize using `CountVectorizer`.
4. Train `MultinomialNB` and evaluate accuracy & confusion matrix.
5. Save artifacts:

joblib.dump(spam_detect_model, "spam_model.joblib")
joblib.dump(cv, "vectorizer.joblib")

text

## 4. App architecture

- `app.py`:
  - Loads `spam_model.joblib` and `vectorizer.joblib`.
  - Applies the same preprocessing as the notebook.
  - Exposes a Gradio interface with a text box and label output.
- `requirements.txt`: runtime dependencies for the Space and local runs.

## 5. Results

- Test accuracy: ~0.98385 on the held‑out test set.
- Confusion matrix (ham vs spam) indicates low false positives on normal messages.

> This project demonstrates an end‑to‑end NLP workflow: data cleaning, feature extraction, model training, evaluation, and deployment to a public web app.

2. Repo cleanup / structure
Organize folders so recruiter ko ek glance mein samajh aa jaye:

text
Spam_classifier/
├─ app.py
├─ requirements.txt
├─ Spam_classifier.ipynb        # main EDA + training
├─ Spamclassifier.py            # optional: CLI / script, warna hata sakta hai
├─ SMSSpamCollection.txt
├─ spam_model.joblib
├─ vectorizer.joblib
├─ readme / README.md           # final README (one of them)
└─ Spam_filter/                 # (optional) original notebook folder, if needed
​

3. Screenshots / demo GIF (optional but strong flex)
Local ya Space UI ka screenshot le:

Input: normal message (ham).

Input: typical spam (offer / link).

Repo mein assets/ folder bana:

bash
mkdir assets
mv path/to/screenshot1.png assets/ui_ham.png
mv path/to/screenshot2.png assets/ui_spam.png
git add assets/*
git commit -m "Add app screenshots"
git push origin main
git push github main
README me embed:

text
## Demo

![Ham prediction](assets/ui_ham.png)
![Spam prediction](assets/ui_spam.png)
4. Cross‑linking HF Space and GitHub
Hugging Face Space description me dal:

“Source code: https://github.com/aryankd123/Spam_classifier”
