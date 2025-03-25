import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

class ProductVisualizationDashboard:
    """Interactive dashboard with multiple views"""
    
    def __init__(self, data, cluster_labels=None):
        self.data = data
        self.cluster_labels = cluster_labels
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
        self._setup_layout()
        self._register_callbacks()
        
    def _setup_layout(self):
        """Configure dashboard layout"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col(html.H1("Advanced Product Analytics"), width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='product-selector',
                        options=[{'label': p, 'value': p} 
                                for p in self.data['Product Name'].unique()],
                        value=self.data['Product Name'].iloc[0]
                    )
                ], width=6),
                dbc.Col([
                    dcc.Dropdown(
                        id='metric-selector',
                        options=[
                            {'label': 'Price', 'value': 'Price'},
                            {'label': 'Stock Quantity', 'value': 'Stock Quantity'},
                            {'label': 'Ratings', 'value': 'Product Ratings'}
                        ],
                        value='Price',
                        multi=True
                    )
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='time-series-plot'), width=12)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='cluster-plot'), width=6),
                dbc.Col(dcc.Graph(id='correlation-heatmap'), width=6)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='sentiment-distribution'), width=12)
            ])
        ], fluid=True)
        
    def _register_callbacks(self):
        """Define interactive callbacks"""
        @self.app.callback(
            Output('time-series-plot', 'figure'),
            [Input('product-selector', 'value'),
             Input('metric-selector', 'value')]
        )
        def update_time_series(product, metrics):
            if not isinstance(metrics, list):
                metrics = [metrics]
                
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            product_data = self.data[self.data['Product Name'] == product]
            
            for metric in metrics:
                fig.add_trace(
                    go.Scatter(
                        x=product_data['Manufacturing Date'],
                        y=product_data[metric],
                        name=metric,
                        mode='lines+markers'
                    ),
                    secondary_y=False
                )
                
            # Add cluster information if available
            if self.cluster_labels is not None:
                fig.add_trace(
                    go.Scatter(
                        x=product_data['Manufacturing Date'],
                        y=self.cluster_labels[self.data['Product Name'] == product],
                        name='Cluster',
                        mode='markers',
                        marker=dict(size=10, opacity=0.7)
                    ),
                    secondary_y=True
                )
                
            fig.update_layout(
                title=f'Time Series Analysis for {product}',
                xaxis_title='Date',
                hovermode='x unified'
            )
            return fig
            
        # Additional callbacks for other visualizations...
        
    def run(self, debug=True, port=8050):
        """Start the dashboard server"""
        self.app.run_server(debug=debug, port=port)
