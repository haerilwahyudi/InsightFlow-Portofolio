import nltk
from transformers import pipeline
from textblob import TextBlob
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
class AdvancedSentimentAnalyzer:
    """Multi-method sentiment analysis with visualization"""
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.transformer_pipeline = pipeline(
            "sentiment-analysis",
            model="finiteautomata/bertweet-base-sentiment-analysis"
        )
        
    def analyze(self, text):
        """Comprehensive sentiment analysis"""
        results = {
            'transformer': self._transformer_analysis(text),
            'textblob': self._textblob_analysis(text),
            'spacy': self._spacy_analysis(text),
            'keywords': self._extract_keywords(text)
        }
        results['composite_score'] = self._calculate_composite(results)
        return results
        
    def _transformer_analysis(self, text):
        """State-of-the-art transformer model"""
        result = self.transformer_pipeline(text)[0]
        return {
            'label': result['label'],
            'score': result['score'],
            'sentiment': 1 if result['label'] == 'POS' else -1
        }
        
    def _textblob_analysis(self, text):
        """Traditional sentiment analysis"""
        analysis = TextBlob(text)
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity
        }
        
    def _spacy_analysis(self, text):
        """Entity-aware sentiment analysis"""
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return {
            'entities': entities,
            'noun_phrases': [chunk.text for chunk in doc.noun_chunks]
        }
        
    def _extract_keywords(self, text, n=5):
        """Extract important keywords"""
        doc = self.nlp(text)
        keywords = [
            token.text for token in doc 
            if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'ADJ']
        ]
        return list(set(keywords))[:n]
        
    def generate_wordcloud(self, texts, output_path=None):
        """Visualize frequent terms"""
        wordcloud = WordCloud(width=800, height=400).generate(' '.join(texts))
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        if output_path:
            plt.savefig(output_path)
        return plt.gcf()
        
    def _calculate_composite(self, results):
        """Combine multiple sentiment scores"""
        transformer_score = results['transformer']['sentiment'] * results['transformer']['score']
        textblob_score = results['textblob']['polarity']
        return (transformer_score + textblob_score) / 2
