import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- New: Advanced Text Preprocessing ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    
    # Tokenize, lemmatize, and remove stop words
    tokens = text.split()
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens 
        if word not in stop_words and len(word) > 2
    ]
    
    return " ".join(processed_tokens)
# --- End of New Preprocessing ---

# 1. Load Data
df = pd.read_csv("./dataset/UpdatedResumeDataSet.csv")
df.dropna(inplace=True)

# 2. Preprocess all resume text
print("Preprocessing text... This may take a minute.")
df['Processed_Resume'] = df['Resume'].apply(preprocess_text)
print("Preprocessing complete.")

# 3. Define Features (X) and Target (y)
X = df['Processed_Resume']
y = df['Category']

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Vectorize
# --- Upgraded: Added ngram_range ---
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("Vectorization complete.")

# 6. Train Model
# --- Upgraded: Using LinearSVC ---
model = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
model.fit(X_train_vec, y_train)
print("Model training complete.")

# 7. Evaluate Model
y_pred = model.predict(X_test_vec)
print("\n--- Model Evaluation ---")
print(classification_report(y_test, y_pred))
print("------------------------\n")

# 8. Save
joblib.dump(vectorizer, './vectorizer/vectorizer.pkl')
joblib.dump(model, './model/resume_classifier.pkl')

print("New model and vectorizer saved successfully!")