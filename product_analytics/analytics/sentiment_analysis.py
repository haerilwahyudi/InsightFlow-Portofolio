import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline

class ProductSentimentAnalyzer:
    def __init__(self):
        nltk.download('vader_lexicon')
        self.sia = SentimentIntensityAnalyzer()
        self.transformer_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    
    def analyze_vader(self, text):
        """Basic sentiment analysis using VADER"""
        return self.sia.polarity_scores(text)
    
    def analyze_textblob(self, text):
        """Alternative sentiment scoring"""
        analysis = TextBlob(text)
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity
        }
    
    def analyze_transformers(self, text):
        """Advanced transformer-based analysis"""
        return self.transformer_pipeline(text)[0]
    
    def batch_analyze_reviews(self, review_df):
        """Process dataframe of product reviews"""
        results = []
        for _, row in review_df.iterrows():
            analysis = {
                'product_id': row['Product ID'],
                'vader': self.analyze_vader(row['Review']),
                'textblob': self.analyze_textblob(row['Review']),
                'transformer': self.analyze_transformers(row['Review'])
            }
            results.append(analysis)
        return pd.DataFrame(results) 
