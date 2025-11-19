"""
Fake Job Posting Classifier - Streamlit Application
Deploys a machine learning model to classify job postings as real or fake.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Fake Job Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK resources (cached)
@st.cache_resource
def download_nltk_resources():
    """Download required NLTK resources for text preprocessing."""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK resources: {e}")
        return False

# Initialize NLTK resources
download_nltk_resources()

# Text preprocessing function
@st.cache_data
def get_preprocessing_components():
    """Get preprocessing components (stopwords and lemmatizer)."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

def clean_text(text, stop_words, lemmatizer):
    """
    Clean and preprocess text data.
    
    Args:
        text: Input text string
        stop_words: Set of stopwords to remove
        lemmatizer: WordNetLemmatizer instance
        
    Returns:
        Cleaned text string
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    
    # Remove punctuation
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    
    # Remove digits
    text = re.sub(r"\d+", "", text)
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Lemmatize and remove stopwords
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    
    return " ".join(words)

@st.cache_resource
def train_model(dataset_path):
    """
    Train the LinearSVC model on the dataset.
    
    Args:
        dataset_path: Path to the CSV file containing job postings
        
    Returns:
        tuple: (trained_model, tfidf_vectorizer, accuracy, classification_report_text)
    """
    try:
        # Load dataset
        df = pd.read_csv(dataset_path, encoding="utf-8", on_bad_lines="skip")
        
        # Basic cleaning
        df = df.drop_duplicates()
        df = df.dropna(subset=['fraudulent'])
        df['title'] = df['title'].fillna('')
        df['description'] = df['description'].fillna('')
        df['requirements'] = df['requirements'].fillna('')
        
        # Combine text fields
        df['text'] = df['title'] + ' ' + df['description'] + ' ' + df['requirements']
        
        # Prepare features and target
        X = df['text']
        y = df['fraudulent']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Get preprocessing components
        stop_words, lemmatizer = get_preprocessing_components()
        
        # Clean text
        X_train_clean = X_train.apply(lambda x: clean_text(x, stop_words, lemmatizer))
        X_test_clean = X_test.apply(lambda x: clean_text(x, stop_words, lemmatizer))
        
        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        X_train_tfidf = tfidf.fit_transform(X_train_clean)
        X_test_tfidf = tfidf.transform(X_test_clean)
        
        # Train LinearSVC model
        svm_model = LinearSVC(random_state=42)
        svm_model.fit(X_train_tfidf, y_train)
        
        # Evaluate model
        y_pred = svm_model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return svm_model, tfidf, accuracy, report, len(X_train), len(X_test)
        
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, None, None, None, None

def predict_job_posting(text, model, tfidf, stop_words, lemmatizer):
    """
    Predict if a job posting is fake or real.
    
    Args:
        text: Job posting text
        model: Trained LinearSVC model
        tfidf: Trained TF-IDF vectorizer
        stop_words: Set of stopwords
        lemmatizer: WordNetLemmatizer instance
        
    Returns:
        tuple: (prediction, confidence_score)
    """
    # Clean the input text
    cleaned_text = clean_text(text, stop_words, lemmatizer)
    
    # Transform using TF-IDF
    text_tfidf = tfidf.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(text_tfidf)[0]
    
    # Get decision function score (distance from hyperplane)
    decision_score = model.decision_function(text_tfidf)[0]
    
    # Convert to probability-like score (using sigmoid approximation)
    confidence = 1 / (1 + np.exp(-decision_score))
    
    return prediction, confidence

# Main Streamlit App
def main():
    """Main application function."""
    
    # Title and header
    st.title("🔍 Fake Job Posting Classifier")
    st.markdown("---")
    st.markdown(
        """
        This application uses a **LinearSVC (Support Vector Machine)** model to classify job postings as **Real** or **Fake**.
        The model was trained on a dataset of job postings and achieves high accuracy in detecting fraudulent listings.
        """
    )
    
    # Sidebar for model training
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        st.markdown("---")
        
        # Dataset upload or path input
        st.subheader("Dataset")
        dataset_option = st.radio(
            "Choose dataset source:",
            ["Upload CSV", "Enter File Path"],
            help="Upload a CSV file or provide the path to your dataset"
        )
        
        dataset_path = None
        
        if dataset_option == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload fake_job_postings.csv",
                type=['csv'],
                help="Upload the CSV file containing job postings with 'fraudulent' column"
            )
            if uploaded_file is not None:
                # Save uploaded file temporarily
                import tempfile
                import os
                temp_dir = tempfile.mkdtemp()
                dataset_path = os.path.join(temp_dir, uploaded_file.name)
                with open(dataset_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
        else:
            dataset_path = st.text_input(
                "Enter dataset path:",
                value="fake_job_postings.csv",
                help="Path to the CSV file (relative or absolute)"
            )
        
        # Train model button
        train_button = st.button("🚀 Train Model", type="primary", use_container_width=True)
        
        # Model info section
        st.markdown("---")
        st.subheader("ℹ️ Model Information")
        st.markdown("""
        - **Algorithm**: LinearSVC (Support Vector Machine)
        - **Features**: TF-IDF Vectorization (10,000 features)
        - **N-grams**: 1-2 (unigrams and bigrams)
        - **Text Preprocessing**: 
          - Lowercase conversion
          - URL removal
          - HTML tag removal
          - Punctuation removal
          - Stopword removal
          - Lemmatization
        """)
        
        # Test examples section
        st.markdown("---")
        with st.expander("🧪 Quick Test Examples"):
            st.markdown("**Real Job Example:**")
            st.code("""
Title: Senior Software Engineer
Description: We are seeking a highly skilled Senior Software Engineer to join our dynamic development team. The ideal candidate will have extensive experience in full-stack development, with a strong background in Python, JavaScript, and cloud technologies.
Requirements: Bachelor's degree in Computer Science, 5+ years of experience, proficiency in Python and JavaScript.
            """, language=None)
            
            st.markdown("**Fake Job Example:**")
            st.code("""
Title: Work From Home - Easy Money - No Experience Needed
Description: EARN $5000 PER WEEK FROM HOME! No experience required! Just work 2 hours a day! Send us $99 processing fee to get started. Contact: scammer@fakeemail.com
Requirements: Must send personal information and processing fee.
            """, language=None)
            
            st.info("💡 See `test_examples.md` for more detailed examples!")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Job Posting")
        st.markdown("Enter the job posting details below:")
        
        # Input form
        with st.form("job_posting_form"):
            job_title = st.text_input(
                "Job Title *",
                placeholder="e.g., Senior Software Engineer",
                help="Enter the job title"
            )
            
            job_description = st.text_area(
                "Job Description *",
                placeholder="Enter the job description...",
                height=200,
                help="Enter the detailed job description"
            )
            
            job_requirements = st.text_area(
                "Requirements",
                placeholder="Enter job requirements (optional)...",
                height=150,
                help="Enter the job requirements (optional)"
            )
            
            submit_button = st.form_submit_button("🔍 Classify Job Posting", use_container_width=True)
    
    with col2:
        st.header("📊 Prediction Results")
        
        # Check if model is trained
        if 'model' not in st.session_state or st.session_state.model is None:
            st.info("👈 Please train the model first using the sidebar.")
            st.markdown("""
            ### How to use:
            1. Upload or specify the path to your dataset in the sidebar
            2. Click "Train Model" to train the classifier
            3. Enter job posting details in the form
            4. Click "Classify Job Posting" to get predictions
            """)
        else:
            # Display model metrics if available
            if 'model_accuracy' in st.session_state and st.session_state.model_accuracy is not None:
                st.metric("Model Accuracy", f"{st.session_state.model_accuracy:.2%}")
            
            # Show prediction area
            prediction_placeholder = st.empty()
            confidence_placeholder = st.empty()
            details_placeholder = st.empty()
    
    # Train model if button clicked
    if train_button and dataset_path:
        with st.spinner("🔄 Training model... This may take a few minutes."):
            model, tfidf, accuracy, report, train_size, test_size = train_model(dataset_path)
            
            if model is not None:
                st.session_state.model = model
                st.session_state.tfidf = tfidf
                st.session_state.model_accuracy = accuracy
                st.session_state.classification_report = report
                st.session_state.train_size = train_size
                st.session_state.test_size = test_size
                
                st.success("✅ Model trained successfully!")
                
                # Display training results
                st.subheader("📈 Training Results")
                col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                with col_metrics1:
                    st.metric("Training Samples", train_size)
                with col_metrics2:
                    st.metric("Test Samples", test_size)
                with col_metrics3:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                
                # Show classification report
                with st.expander("📋 View Classification Report"):
                    st.text(report)
            else:
                st.error("❌ Failed to train model. Please check your dataset.")
    
    # Make prediction if form submitted
    if submit_button and 'model' in st.session_state and st.session_state.model is not None:
        if not job_title or not job_description:
            st.warning("⚠️ Please fill in at least Job Title and Job Description.")
        else:
            # Combine text fields
            combined_text = f"{job_title} {job_description} {job_requirements if job_requirements else ''}"
            
            # Get preprocessing components
            stop_words, lemmatizer = get_preprocessing_components()
            
            # Make prediction
            with st.spinner("🔄 Analyzing job posting..."):
                prediction, confidence = predict_job_posting(
                    combined_text,
                    st.session_state.model,
                    st.session_state.tfidf,
                    stop_words,
                    lemmatizer
                )
            
            # Display results
            with col2:
                prediction_placeholder.empty()
                confidence_placeholder.empty()
                details_placeholder.empty()
                
                # Prediction result
                if prediction == 0:
                    st.success("✅ **REAL JOB POSTING**")
                    st.balloons()
                else:
                    st.error("❌ **FAKE JOB POSTING**")
                    st.warning("⚠️ This job posting appears to be fraudulent. Proceed with caution.")
                
                # Confidence score
                confidence_percent = confidence * 100 if prediction == 1 else (1 - confidence) * 100
                st.metric("Confidence", f"{confidence_percent:.1f}%")
                
                # Details
                with details_placeholder.expander("📋 View Details"):
                    st.markdown(f"**Job Title:** {job_title}")
                    st.markdown(f"**Prediction:** {'Fake' if prediction == 1 else 'Real'}")
                    st.markdown(f"**Confidence Score:** {confidence:.4f}")
                    st.markdown(f"**Decision Function Value:** {st.session_state.model.decision_function(st.session_state.tfidf.transform([clean_text(combined_text, stop_words, lemmatizer)]))[0]:.4f}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Fake Job Posting Classifier | Built with Streamlit & Scikit-learn</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

