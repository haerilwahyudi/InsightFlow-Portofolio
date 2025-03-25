import logging
from data_processing.data_loader import ProductDataLoader
from data_processing.feature_engineering import ProductFeatureEngineer
from analytics.product_analysis import ComprehensiveProductAnalyzer
from visualization.interactive_plots import ProductVisualizationDashboard

def configure_logging():
    """Set up comprehensive logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('product_analytics.log'),
            logging.StreamHandler()
        ]
    )

def main():
    # Configuration
    configure_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Data Loading
        logger.info("Loading product data...")
        loader = ProductDataLoader(config={'csv_encoding': 'utf-8'})
        raw_data = loader.load_data(source_type='csv', filepath='products.csv')
        
        # Feature Engineering
        logger.info("Engineering features...")
        engineer = ProductFeatureEngineer(
            extract_dimensions=True,
            add_time_features=True
        )
        processed_data = engineer.transform(raw_data)
        
        # Comprehensive Analysis
        logger.info("Performing product analysis...")
        analyzer = ComprehensiveProductAnalyzer(processed_data)
        analysis_results = analyzer.run_full_analysis()
        
        # Visualization
        logger.info("Launching dashboard...")
        dashboard = ProductVisualizationDashboard(
            processed_data,
            cluster_labels=analysis_results['cluster_labels']
        )
        dashboard.run(debug=False)
        
    except Exception as e:
        logger.error(f"System error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
