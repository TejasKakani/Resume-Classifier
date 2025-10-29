import joblib
import PyPDF2
import re
import numpy as np  # Keep NumPy
from flask import Flask, request, render_template
from docx import Document

# Load trained model and vectorizer immediately
vectorizer = joblib.load("./vectorizer/vectorizer.pkl")
model = joblib.load("./model/resume_classifier.pkl")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def classify_resume(uploaded_file):
    raw_text = ""
    
    # 1. Extract Text
    if uploaded_file.filename.endswith('.pdf'):
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            raw_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return None, None, None
            
    elif uploaded_file.filename.endswith('.docx'):
        try:
            doc = Document(uploaded_file)
            raw_text = " ".join([p.text for p in doc.paragraphs])
        except Exception as e:
            print(f"Error reading DOCX: {e}")
            return None, None, None
    else:
        return None, None, None

    # 2. Preprocess and Vectorize
    processed_text = preprocess_text(raw_text)
    input_features = vectorizer.transform([processed_text])
    
    # 3. Get Probabilities (This will work now!)
    probabilities = model.predict_proba(input_features)[0]
    
    # 4. Get the category names (classes)
    categories = model.classes_
    
    # 5. Combine names with their probabilities
    prediction_list = list(zip(categories, probabilities))
    
    # 6. Sort by probability (highest first) and take top 5
    sorted_predictions = sorted(prediction_list, key=lambda x: x[1], reverse=True)[:5]
    
    # --- START: Updated Keyword Extraction Logic ---
    
    # 7. Get the top predicted category name
    top_category_name = sorted_predictions[0][0]
    
    # 8. Get the index of that category
    category_index = np.where(model.classes_ == top_category_name)[0][0]
    
    # 9. Get the scores (coefficients) for all words for that category
    # --- CHANGE: Using model.coef_ instead of feature_log_prob_ ---
    feature_scores = model.coef_[category_index]
    
    # 10. Get all word names
    feature_names = vectorizer.get_feature_names_out()
    
    # 11. Sort words by their score (importance) and get top 50
    indices = np.argsort(feature_scores)[-50:]
    top_model_keywords = [feature_names[i] for i in indices]
    
    # 12. Find which of these words are *actually in the resume*
    resume_words = set(processed_text.split()) 
    keywords_found = [word for word in top_model_keywords if word in resume_words]
    
    # --- END: Updated Keyword Extraction Logic ---
    
    return sorted_predictions, keywords_found, raw_text

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    predictions = None
    keywords = None
    resume_text = None  
    
    if request.method == "POST":
        if "resume" in request.files:
            file = request.files["resume"]
            if file and (file.filename.endswith('.pdf') or file.filename.endswith('.docx')):
               predictions, keywords, resume_text = classify_resume(file)
               
    return render_template("index.html", 
                           predictions=predictions, 
                           keywords=keywords, 
                           resume_text=resume_text)

if __name__ == "__main__":
    app.run(debug=True)