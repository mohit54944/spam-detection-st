# Spam Detection NLP Mini-Project with Streamlit UI

import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
st.set_page_config(
    page_title="Spam Detection Tool",
    page_icon="📧",
    layout="centered"
)

# --- Static UI Elements ---
st.title("📧 Spam Detection Tool")
st.markdown("This application uses a **Multinomial Naive Bayes** model to classify messages as Spam or Ham (Not Spam). Enter a message below to see the prediction.")


# --- Setup: Download NLTK data (only needs to be done once) ---
@st.cache_resource
def download_nltk_data():
    try:
        stopwords.words('english')
    except LookupError:
        # Use a spinner for a better user experience during download
        with st.spinner("Downloading necessary NLTK data (stopwords)..."):
            nltk.download('stopwords')

download_nltk_data()

# --- Text Preprocessing Function ---
def preprocess_text(text):

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    # Corrected regex and ensured proper chaining
    text = re.sub('[^a-zA-Z]', ' ', str(text)).lower()
    words = text.split()
    processed_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(processed_words)

# --- Model Training and Loading ---
@st.cache_data(show_spinner="Training model...")
def load_and_train_model(filepath):

    # Load data
    df = pd.read_csv(filepath, encoding='latin-1')
    df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
    df = df.dropna(subset=['message']) # Drop rows where message is missing
    df = df[['label', 'message']]

    # Preprocess messages
    df['processed_message'] = df['message'].apply(preprocess_text)

    # Feature extraction (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df['processed_message']).toarray()
    y = pd.get_dummies(df['label'], drop_first=True).values.ravel()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Train the best model: Multinomial Naive Bayes
    model = MultinomialNB()
    model.fit(X_train, y_train)

    return model, vectorizer, X_test, y_test

# --- Main Application Logic ---
try:
    FILEPATH = 'spam.csv'
    model, vectorizer, X_test, y_test = load_and_train_model(FILEPATH)

    # --- User Input Section ---
    with st.form(key='message_form'):
        user_input = st.text_area("Enter the message you want to classify:", height=150)
        submit_button = st.form_submit_button(label='Classify Message')

    if submit_button and user_input:
        # 1. Preprocess the user's input
        processed_input = preprocess_text(user_input)
        
        # 2. Transform the input using the TF-IDF vectorizer
        vectorized_input = vectorizer.transform([processed_input]).toarray()
        
        # 3. Make a prediction
        prediction = model.predict(vectorized_input)[0]
        
        # 4. Display the result
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("🚨 This message is likely **Spam**.")
        else:
            st.success("✅ This message is likely **Ham (Not Spam)**.")

    # --- Model Performance Section ---
    st.sidebar.title("Model Performance")
    st.sidebar.markdown("The model was evaluated on a test set of unseen data.")
    
    # Calculate metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    st.sidebar.metric("Accuracy", f"{accuracy:.4f}")
    st.sidebar.metric("Precision", f"{precision:.4f}")
    st.sidebar.metric("Recall", f"{recall:.4f}")

    # Display Confusion Matrix
    st.sidebar.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=ax)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.sidebar.pyplot(fig)


except FileNotFoundError:
    st.error(f"Error: Dataset not found at '{FILEPATH}'.")
    st.info("Please make sure you have the `spam.csv` file in the same directory as this script.")

except Exception as e:
    st.error(f"An error occurred during model loading or processing: {e}")

