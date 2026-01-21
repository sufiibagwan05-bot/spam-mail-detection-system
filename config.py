"""
Configuration file for the Spam Email Detection System
Modify these settings to customize the application behavior
"""

# Application Settings
APP_TITLE = "🛡️ Email Spam Detection System"
APP_ICON = "🛡️"
PAGE_LAYOUT = "wide"

# Model Settings
MODEL_FILE = "spam_model.pkl"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# TF-IDF Vectorizer Settings
TFIDF_CONFIG = {
    'stop_words': 'english',
    'max_features': 5000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'max_df': 0.95
}

# Naive Bayes Settings
NB_CONFIG = {
    'alpha': 0.1
}

# UI Settings
CONFIDENCE_THRESHOLD = 0.8
MAX_HISTORY_ITEMS = 100
CHART_HEIGHT = 300

# Color Scheme
COLORS = {
    'spam': '#ff6b6b',
    'ham': '#2ed573',
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'background': '#f8f9fa'
}

# Sample Test Messages
TEST_MESSAGES = {
    "🎁 Promotional Spam": "Congratulations! You've won $1,000,000! Click here to claim your prize now! Limited time offer!",
    "🏦 Phishing Attempt": "URGENT: Your bank account has been compromised. Verify your details immediately to avoid closure.",
    "💼 Legitimate Business": "Hi John, I hope this email finds you well. I wanted to follow up on our meeting yesterday regarding the project timeline.",
    "👥 Personal Message": "Hey! Are we still on for dinner tonight? Let me know if you need me to pick anything up on the way.",
    "📰 Newsletter": "Thank you for subscribing to our newsletter. Here are this week's top technology news and updates.",
    "🚨 Urgent Scam": "FINAL NOTICE: Your warranty is about to expire! Click here to renew now and avoid losing coverage!",
    "📧 Work Email": "Could you please review the attached document and provide your feedback by end of day Friday?",
    "💰 Money Scam": "You've been pre-approved for a $50,000 loan! No credit check required! Apply now!",
    "📅 Meeting Reminder": "Reminder: Team meeting tomorrow at 2 PM in conference room A. Please bring your quarterly reports.",
    "🎯 Fake Offer": "Click here to claim your free iPhone 15! Limited quantities available! Hurry, offer expires soon!"
}

# Feature Importance Settings
TOP_FEATURES_COUNT = 20

# Batch Processing Settings
MAX_BATCH_SIZE = 10000
SUPPORTED_FILE_TYPES = ['csv']

# Performance Settings
PREDICTION_TIMEOUT = 30  # seconds
MAX_MESSAGE_LENGTH = 10000  # characters

# Logging Settings
LOG_PREDICTIONS = True
LOG_FILE = "predictions.log"

# Security Settings
SANITIZE_INPUT = True
MAX_UPLOAD_SIZE = 10  # MB