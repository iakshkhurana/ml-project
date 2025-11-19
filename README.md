# Fake Job Posting Classifier - Streamlit App

A machine learning application that classifies job postings as **Real** or **Fake** using a LinearSVC (Support Vector Machine) model.

## Features

- 🔍 **Real-time Classification**: Input job posting details and get instant predictions
- 📊 **High Accuracy**: Model achieves ~98% accuracy on test data
- 🧠 **Advanced NLP**: Uses TF-IDF vectorization with text preprocessing
- 🎨 **User-Friendly Interface**: Clean and intuitive Streamlit UI
- ⚡ **Fast Predictions**: Cached model training for quick startup

## Model Details

- **Algorithm**: LinearSVC (Linear Support Vector Classifier)
- **Features**: TF-IDF Vectorization (10,000 features)
- **N-grams**: 1-2 (unigrams and bigrams)
- **Text Preprocessing**:
  - Lowercase conversion
  - URL removal
  - HTML tag removal
  - Punctuation removal
  - Stopword removal
  - Lemmatization

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download NLTK resources** (automatically done on first run):
   - stopwords
   - wordnet

## Usage

### Running the Application

1. **Start the Streamlit app**:
```bash
streamlit run app.py
```

2. **Train the Model**:
   - In the sidebar, either:
     - Upload your `fake_job_postings.csv` file, OR
     - Enter the path to your dataset file
   - Click "🚀 Train Model" button
   - Wait for training to complete (may take a few minutes)

3. **Classify Job Postings**:
   - Enter job title (required)
   - Enter job description (required)
   - Enter requirements (optional)
   - Click "🔍 Classify Job Posting"
   - View the prediction results

### Test Examples

See `test_examples.md` for ready-to-use test cases including:
- ✅ **Real Job Postings**: Software Engineer, Data Scientist, Marketing Manager
- ❌ **Fake Job Postings**: Work-from-home scams, pyramid schemes, suspicious remote jobs

Copy and paste these examples directly into the Streamlit app to test the classifier!

### Dataset Format

The CSV file should contain the following columns:
- `title`: Job title
- `description`: Job description
- `requirements`: Job requirements (optional)
- `fraudulent`: Target variable (0 = Real, 1 = Fake)

Example dataset structure:
```csv
job_id,title,description,requirements,fraudulent
1,Software Engineer,We are looking for...,Bachelor's degree...,0
2,Data Scientist,Join our team...,Master's degree...,1
```

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── sol.ipynb            # Original Jupyter notebook with model development
```

## Model Performance

The model achieves the following performance metrics:
- **Accuracy**: ~98.38%
- **Precision (Fake)**: ~98%
- **Recall (Fake)**: ~68%
- **F1-Score (Fake)**: ~80%

## Technical Details

### Text Preprocessing Pipeline

1. **Lowercase Conversion**: Converts all text to lowercase
2. **URL Removal**: Removes HTTP/HTTPS links
3. **HTML Tag Removal**: Strips HTML tags
4. **Punctuation Removal**: Removes punctuation marks
5. **Digit Removal**: Removes numeric characters
6. **Whitespace Normalization**: Removes extra spaces
7. **Stopword Removal**: Removes common English stopwords
8. **Lemmatization**: Converts words to their root form

### Model Training

- **Train-Test Split**: 80% training, 20% testing
- **Stratification**: Maintains class distribution
- **Random State**: 42 (for reproducibility)
- **TF-IDF Parameters**:
  - Max features: 10,000
  - N-gram range: (1, 2)
  - Sublinear TF: True

## Troubleshooting

### Common Issues

1. **NLTK Download Errors**:
   - The app automatically downloads NLTK resources on first run
   - If issues persist, manually download:
     ```python
     import nltk
     nltk.download('stopwords')
     nltk.download('wordnet')
     ```

2. **Dataset Not Found**:
   - Ensure the CSV file path is correct
   - Check that the file has the required columns

3. **Memory Issues**:
   - Large datasets may require significant RAM
   - Consider using a smaller subset for testing

## Future Enhancements

- [ ] Save/load trained models to avoid retraining
- [ ] Batch prediction for multiple job postings
- [ ] Model comparison dashboard
- [ ] Feature importance visualization
- [ ] API endpoint for programmatic access

## License

This project is for educational purposes.

## Author

Developed as part of Machine Learning project work.

---

**Note**: This application is designed for educational and demonstration purposes. Always verify job postings through official channels before applying.

