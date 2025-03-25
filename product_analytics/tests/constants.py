# Data processing constants
DATE_FORMAT = '%Y-%m-%d'
DEFAULT_TIME_FREQ = 'W'  # Weekly frequency

# Model defaults
DEFAULT_CLUSTERS = 4
FORECAST_HORIZON = 90  # days

# Visualization
COLOR_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
]

# API Keys (would be environment variables in production)
SENTIMENT_API_KEY = None
DATABASE_CREDENTIALS = {
    'host': 'localhost',
    'user': 'admin',
    'password': 'securepassword',
    'database': 'product_analytics'
}
