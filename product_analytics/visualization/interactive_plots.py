 import plotly.express as px
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

class ProductAnalyticsDashboard:
    def __init__(self, df):
        self.df = df
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
    def create_time_series_plot(self):
        """Interactive time-series visualization"""
        return px.line(
            self.df,
            x='Date',
            y='Stock Quantity',
            color='Product Name',
            title='Product Demand Over Time',
            hover_data=['Price', 'Product Ratings']
        )
    
    def create_sentiment_radar(self, sentiment_df):
        """Radar chart for sentiment analysis"""
        return px.line_polar(
            sentiment_df,
            r='score',
            theta='metric',
            color='product_id',
            line_close=True,
            template='plotly_dark'
        )
    
    def run_dashboard(self):
        """Launch interactive Dash app"""
        self.app.layout = html.Div([
            html.H1("Product Analytics Dashboard"),
            dcc.Graph(figure=self.create_time_series_plot()),
            dcc.Dropdown(
                id='product-selector',
                options=[{'label': p, 'value': p} for p in self.df['Product Name'].unique()],
                value=self.df['Product Name'].iloc[0]
            ),
            html.Div(id='product-details')
        ])
        
        self.app.run_server(debug=True)
