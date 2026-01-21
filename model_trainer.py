"""
Enhanced Spam Email Detection Model
Capstone Project - Email Security System
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import pickle
import os
from datetime import datetime

class SpamDetector:
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.training_history = {}
    
    def load_data(self, file_path=None):
        """Load training data from file or create sample data"""
        if file_path and os.path.exists(file_path):
            try:
                data = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'])
                data['label'] = data['label'].map({'ham': 0, 'spam': 1})
                return data
            except:
                print("Error loading data file. Using sample data instead.")
        
        # Create enhanced sample data for demonstration
        sample_data = {
            'message': [
                # Spam examples
                "Congratulations! You've won $1000! Click here now!",
                "URGENT: Your account will be closed. Verify now!",
                "FREE MONEY! No strings attached! Act now!",
                "You have been selected for a special offer!",
                "WINNER! Claim your prize immediately!",
                "Get rich quick! Make $5000 per week working from home!",
                "Your credit card has been charged $500. Click to dispute.",
                "Hot singles in your area want to meet you!",
                "Lose 30 pounds in 30 days with this miracle pill!",
                "Nigerian prince needs your help transferring millions!",
                "FINAL NOTICE: Your warranty is about to expire!",
                "You've been pre-approved for a $50,000 loan!",
                "Click here to claim your free iPhone 15!",
                "BREAKING: Local mom discovers weight loss secret!",
                "Your computer is infected! Download our antivirus now!",
                
                # Ham (legitimate) examples
                "Hi, how are you doing today?",
                "Meeting scheduled for tomorrow at 3 PM",
                "Can you send me the report by Friday?",
                "Thanks for your help with the project",
                "Let's catch up over coffee sometime",
                "The quarterly results look promising this year",
                "Please review the attached document and provide feedback",
                "Reminder: Team meeting at 2 PM in conference room A",
                "Happy birthday! Hope you have a wonderful day",
                "The presentation went well. Thanks for your support",
                "Could you please update the client database?",
                "Looking forward to our collaboration on this project",
                "The new policy changes will take effect next month",
                "Please confirm your attendance for the workshop",
                "Thank you for your prompt response to my inquiry"
            ] * 50,  # Multiply to create more training samples
            'label': [1] * 15 * 50 + [0] * 15 * 50  # 15 spam + 15 ham, repeated 50 times
        }
        
        return pd.DataFrame(sample_data)
    
    def train_model(self, data_path=None):
        """Train the spam detection model"""
        print("Loading training data...")
        data = self.load_data(data_path)
        
        print(f"Dataset loaded: {len(data)} samples")
        print(f"Spam messages: {sum(data['label'])}")
        print(f"Ham messages: {len(data) - sum(data['label'])}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            data['message'], data['label'], 
            test_size=0.2, 
            random_state=42,
            stratify=data['label']
        )
        
        print("Training model...")
        
        # Create and train the pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english',
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )),
            ('clf', MultinomialNB(alpha=0.1))
        ])
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Store training history
        self.training_history = {
            'timestamp': datetime.now(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        print("\n" + "="*50)
        print("MODEL TRAINING COMPLETED")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
        
        return accuracy, precision, recall
    
    def predict(self, message):
        """Predict if a message is spam"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        prediction = self.model.predict([message])[0]
        probability = self.model.predict_proba([message])[0]
        confidence = max(probability)
        
        return {
            'prediction': int(prediction),
            'label': 'Spam' if prediction == 1 else 'Ham',
            'confidence': confidence,
            'spam_probability': probability[1],
            'ham_probability': probability[0]
        }
    
    def predict_batch(self, messages):
        """Predict multiple messages at once"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        predictions = self.model.predict(messages)
        probabilities = self.model.predict_proba(messages)
        
        results = []
        for i, message in enumerate(messages):
            results.append({
                'message': message,
                'prediction': int(predictions[i]),
                'label': 'Spam' if predictions[i] == 1 else 'Ham',
                'confidence': max(probabilities[i]),
                'spam_probability': probabilities[i][1],
                'ham_probability': probabilities[i][0]
            })
        
        return results
    
    def save_model(self, filepath='spam_model.pkl'):
        """Save the trained model to disk"""
        if not self.is_trained or self.model is None:
            raise ValueError("No trained model to save.")
        
        model_data = {
            'model': self.model,
            'training_history': self.training_history,
            'timestamp': datetime.now()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='spam_model.pkl'):
        """Load a trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found.")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.training_history = model_data.get('training_history', {})
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
        if self.training_history:
            print(f"Model accuracy: {self.training_history.get('accuracy', 'N/A'):.4f}")
    
    def get_feature_importance(self, top_n=20):
        """Get the most important features for spam detection"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained.")
        
        # Get feature names and importance scores
        feature_names = self.model.named_steps['tfidf'].get_feature_names_out()
        feature_log_prob = self.model.named_steps['clf'].feature_log_prob_
        
        # Calculate feature importance (difference between spam and ham log probabilities)
        spam_features = feature_log_prob[1] - feature_log_prob[0]
        
        # Get top spam indicators
        top_spam_indices = np.argsort(spam_features)[-top_n:][::-1]
        top_spam_features = [(feature_names[i], spam_features[i]) for i in top_spam_indices]
        
        # Get top ham indicators
        top_ham_indices = np.argsort(spam_features)[:top_n]
        top_ham_features = [(feature_names[i], spam_features[i]) for i in top_ham_indices]
        
        return {
            'spam_indicators': top_spam_features,
            'ham_indicators': top_ham_features
        }

def main():
    """Main function to train and test the model"""
    detector = SpamDetector()
    
    # Train the model
    print("Starting model training...")
    accuracy, precision, recall = detector.train_model()
    
    # Save the model
    detector.save_model()
    
    # Test with sample messages
    test_messages = [
        "Congratulations! You won a lottery! Click here to claim!",
        "Hi John, can we schedule a meeting for tomorrow?",
        "URGENT: Your account needs verification immediately!",
        "Thanks for the great presentation yesterday."
    ]
    
    print("\n" + "="*50)
    print("TESTING MODEL WITH SAMPLE MESSAGES")
    print("="*50)
    
    for message in test_messages:
        result = detector.predict(message)
        print(f"\nMessage: {message}")
        print(f"Prediction: {result['label']} (Confidence: {result['confidence']:.3f})")
    
    # Show feature importance
    print("\n" + "="*50)
    print("TOP SPAM INDICATORS")
    print("="*50)
    
    features = detector.get_feature_importance(10)
    for feature, score in features['spam_indicators']:
        print(f"{feature}: {score:.3f}")

if __name__ == "__main__":
    main()