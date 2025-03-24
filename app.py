import joblib
import PyPDF2
import re
from flask import Flask, request
from flask import render_template

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def classify_resume(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    processed_text = preprocess_text(text)
    input_features = vectorizer.transform([processed_text])
    prediction = model.predict(input_features)[0]
    return prediction

# Load trained model and vectorizer
vectorizer = joblib.load("./vectorizer/vectorizer.pkl")
model = joblib.load("./model/resume_classifier.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    job_category = None
    if request.method == "POST":
        file = request.files["resume"]
        if file:
            job_category = classify_resume(file)
    return render_template("index.html", job_category=job_category)