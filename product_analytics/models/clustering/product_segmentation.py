 
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import numpy as np

class ProductClusterAnalyzer:
    """Advanced clustering for product segmentation"""
    
    def __init__(self, data, n_clusters_range=(2, 10)):
        self.data = data
        self.n_clusters_range = n_clusters_range
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(data)
        
    def find_optimal_clusters(self):
        """Determine best number of clusters using multiple methods"""
        results = {}
        
        # Elbow method
        distortions = []
        for k in range(*self.n_clusters_range):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(self.scaled_data)
            distortions.append(kmeans.inertia_)
        results['elbow'] = distortions
        
        # Silhouette analysis
        silhouette_scores = []
        for k in range(*self.n_clusters_range):
            kmeans = KMeans(n_clusters=k)
            labels = kmeans.fit_predict(self.scaled_data)
            silhouette_scores.append(silhouette_score(self.scaled_data, labels))
        results['silhouette'] = silhouette_scores
        
        # Bayesian Information Criterion for GMM
        bic_scores = []
        for k in range(*self.n_clusters_range):
            gmm = GaussianMixture(n_components=k)
            gmm.fit(self.scaled_data)
            bic_scores.append(gmm.bic(self.scaled_data))
        results['bic'] = bic_scores
        
        return results
        
    def cluster_products(self, method='kmeans', **kwargs):
        """Apply selected clustering algorithm"""
        if method == 'kmeans':
            model = KMeans(n_clusters=kwargs.get('n_clusters', 4))
        elif method == 'dbscan':
            model = DBSCAN(
                eps=kwargs.get('eps', 0.5),
                min_samples=kwargs.get('min_samples', 5)
            )
        elif method == 'optics':
            model = OPTICS(
                min_samples=kwargs.get('min_samples', 5),
                xi=kwargs.get('xi', 0.05)
            )
        elif method == 'gmm':
            model = GaussianMixture(
                n_components=kwargs.get('n_components', 4),
                covariance_type=kwargs.get('covariance_type', 'full')
            )
        else:
            raise ValueError("Unsupported clustering method")
            
        labels = model.fit_predict(self.scaled_data)
        return labels
        
    def analyze_clusters(self, labels):
        """Statistical analysis of each cluster"""
        self.data['Cluster'] = labels
        cluster_stats = self.data.groupby('Cluster').agg({
            'Price': ['mean', 'std', 'min', 'max'],
            'Stock Quantity': ['mean', 'sum'],
            'Product Ratings': ['mean', 'count']
        })
        return cluster_stats
