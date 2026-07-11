"""
Main script to orchestrate all time series forecasting models.
Runs KNN, LSTM, and Seq2Seq CNN models sequentially and compares their performance.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from data_preprocessing import WikiTrafficPreprocessor
from evaluation import TimeSeriesEvaluator
from knn_model import train_knn_model
from lstm_model import train_lstm_model
from cnn_seq2seq import train_seq2seq_cnn_model


class WikiTrafficForecastingPipeline:
    """Main pipeline for Wikipedia traffic forecasting."""
    
    def __init__(self, data_path=None, sample_size=5000, results_dir="results"):
        """
        Initialize the forecasting pipeline.
        
        Args:
            data_path (str): Path to the dataset file
            sample_size (int): Sample size for synthetic data
            results_dir (str): Directory to save results
        """
        self.data_path = data_path
        self.sample_size = sample_size
        self.results_dir = results_dir
        self.preprocessor = WikiTrafficPreprocessor()
        self.evaluator = TimeSeriesEvaluator()
        self.results = {}
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
    def load_and_preprocess_data(self):
        """Load and preprocess the data."""
        print("="*80)
        print("STEP 1: DATA LOADING AND PREPROCESSING")
        print("="*80)
        
        # Preprocess data
        result = self.preprocessor.preprocess_pipeline(
            file_path=self.data_path,
            sample_size=self.sample_size
        )
        
        self.data = result['data']
        self.feature_columns = result['feature_columns']
        
        print(f"\nData preprocessing completed!")
        print(f"Final dataset shape: {self.data.shape}")
        print(f"Number of features: {len(self.feature_columns)}")
        print(f"Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        
        return result
    
    def train_knn_model(self):
        """Train and evaluate KNN model."""
        print("\n" + "="*80)
        print("STEP 2: K-NEAREST NEIGHBORS (KNN) MODEL")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # Split data for KNN
            X_train, X_test, y_train, y_test = self.preprocessor.split_data(self.data)
            
            # Train KNN model
            knn_model, knn_predictions, knn_metrics = train_knn_model(
                X_train, y_train, X_test, y_test, self.feature_columns
            )
            
            # Store results
            self.results['KNN'] = {
                'model': knn_model,
                'predictions': knn_predictions,
                'metrics': knn_metrics,
                'y_test': y_test,
                'training_time': time.time() - start_time
            }
            
            print(f"KNN training completed in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"KNN model failed: {str(e)}")
            self.results['KNN'] = None
    
    def train_lstm_model(self):
        """Train and evaluate LSTM model."""
        print("\n" + "="*80)
        print("STEP 3: LONG SHORT-TERM MEMORY (LSTM) MODEL")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # Train LSTM model
            lstm_model, lstm_predictions, lstm_metrics = train_lstm_model(
                self.data, lookback=30
            )
            
            if lstm_model is not None:
                # Store results
                self.results['LSTM'] = {
                    'model': lstm_model,
                    'predictions': lstm_predictions,
                    'metrics': lstm_metrics,
                    'training_time': time.time() - start_time
                }
                
                print(f"LSTM training completed in {time.time() - start_time:.2f} seconds")
            else:
                print("LSTM model skipped (TensorFlow not available)")
                self.results['LSTM'] = None
                
        except Exception as e:
            print(f"LSTM model failed: {str(e)}")
            self.results['LSTM'] = None
    
    def train_seq2seq_cnn_model(self):
        """Train and evaluate Seq2Seq CNN model."""
        print("\n" + "="*80)
        print("STEP 4: SEQUENCE-TO-SEQUENCE CAUSAL CNN MODEL")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # Train Seq2Seq CNN model
            cnn_model, cnn_predictions, cnn_metrics = train_seq2seq_cnn_model(
                self.data, input_length=64, output_length=32
            )
            
            if cnn_model is not None:
                # Store results
                self.results['Seq2Seq CNN'] = {
                    'model': cnn_model,
                    'predictions': cnn_predictions,
                    'metrics': cnn_metrics,
                    'training_time': time.time() - start_time
                }
                
                print(f"Seq2Seq CNN training completed in {time.time() - start_time:.2f} seconds")
            else:
                print("Seq2Seq CNN model skipped (TensorFlow not available)")
                self.results['Seq2Seq CNN'] = None
                
        except Exception as e:
            print(f"Seq2Seq CNN model failed: {str(e)}")
            self.results['Seq2Seq CNN'] = None
    
    def compare_models(self):
        """Compare all models and generate comparison results."""
        print("\n" + "="*80)
        print("STEP 5: MODEL COMPARISON AND EVALUATION")
        print("="*80)
        
        if not self.results:
            print("No models have been trained yet!")
            return
        
        # Extract metrics for comparison
        comparison_metrics = {}
        for model_name, result in self.results.items():
            if result is not None:
                comparison_metrics[model_name] = result['metrics']
        
        # Print comparison table
        self.print_comparison_table(comparison_metrics)
        
        return comparison_metrics
    
    def print_comparison_table(self, comparison_metrics):
        """Print formatted comparison table."""
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        
        if not comparison_metrics:
            print("No successful model results to compare.")
            return
        
        # Create comparison DataFrame
        df = pd.DataFrame(comparison_metrics).T
        
        # Select key metrics for display
        display_metrics = ['smape', 'mae', 'rmse', 'r2', 'mae_exp', 'rmse_exp']
        available_metrics = [m for m in display_metrics if m in df.columns]
        df_display = df[available_metrics]
        
        print("\nDetailed Metrics:")
        print(df_display.round(4))
        
        print(f"\nSMAPE Comparison (Lower is Better):")
        print("-" * 50)
        smape_scores = df['smape'].sort_values()
        for i, (model, score) in enumerate(smape_scores.items(), 1):
            print(f"{i}. {model:15s}: {score:8.4f}")
        
        print(f"\nTraining Time Comparison:")
        print("-" * 50)
        for model_name, result in self.results.items():
            if result is not None:
                training_time = result['training_time']
                print(f"{model_name:15s}: {training_time:8.2f} seconds")
        
        # Find best model
        if len(smape_scores) > 0:
            best_model = smape_scores.index[0]
            best_score = smape_scores.iloc[0]
            
            print(f"\n🏆 BEST MODEL: {best_model}")
            print(f"   SMAPE Score: {best_score:.4f}")
        
        print("\n" + "="*80)
    
    def run_complete_pipeline(self):
        """Run the complete forecasting pipeline."""
        print("🚀 STARTING WIKIPEDIA TRAFFIC FORECASTING PIPELINE")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results directory: {self.results_dir}")
        
        total_start_time = time.time()
        
        try:
            # Step 1: Data preprocessing
            self.load_and_preprocess_data()
            
            # Step 2: Train KNN model
            self.train_knn_model()
            
            # Step 3: Train LSTM model
            self.train_lstm_model()
            
            # Step 4: Train Seq2Seq CNN model
            self.train_seq2seq_cnn_model()
            
            # Step 5: Compare models
            self.compare_models()
            
            print("\n✅ Pipeline execution completed successfully!")
            print("📁 Check the 'results' directory for detailed results and plots.")
            
            total_time = time.time() - total_start_time
            print(f"Total execution time: {total_time:.2f} seconds")
            
        except Exception as e:
            print(f"\n❌ ERROR: Pipeline failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        return True


def test_imports_and_basic_functionality():
    """Test basic imports and functionality without training models."""
    print("🧪 RUNNING HEALTH CHECK")
    print("=" * 60)
    
    try:
        # Test imports
        print("Testing imports...")
        from data_preprocessing import WikiTrafficPreprocessor
        from evaluation import TimeSeriesEvaluator
        from knn_model import train_knn_model
        from lstm_model import train_lstm_model
        from cnn_seq2seq import train_seq2seq_cnn_model
        print("✅ All imports successful")
        
        # Test data preprocessing
        print("\nTesting data preprocessing...")
        preprocessor = WikiTrafficPreprocessor()
        result = preprocessor.preprocess_pipeline(sample_size=100)  # Small sample for test
        print(f"✅ Data preprocessing successful - Shape: {result['data'].shape}")
        
        # Test evaluator
        print("\nTesting evaluation module...")
        evaluator = TimeSeriesEvaluator()
        print("✅ Evaluation module loaded")
        
        # Test TensorFlow availability
        print("\nTesting TensorFlow availability...")
        try:
            import tensorflow as tf
            print(f"✅ TensorFlow {tf.__version__} available")
        except ImportError:
            print("⚠️  TensorFlow not available - LSTM and CNN models will be skipped")
        
        print("\n🎉 HEALTH CHECK PASSED - All components ready!")
        return True
        
    except Exception as e:
        print(f"\n❌ HEALTH CHECK FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run the forecasting pipeline."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Wikipedia Traffic Time Series Forecasting')
    parser.add_argument('--test', action='store_true', help='Run health check without training models')
    args = parser.parse_args()
    
    # If test flag is provided, run health check only
    if args.test:
        success = test_imports_and_basic_functionality()
        if success:
            print("\n✅ System ready for training! Run without --test flag to train models.")
        else:
            print("\n❌ System has issues. Please fix the errors above.")
        return
    
    # Configuration
    DATA_PATH = "data/validation_score.csv"  # Path to the validation score dataset
    SAMPLE_SIZE = 5000  # Number of time series to use from the dataset
    RESULTS_DIR = "results"
    
    print("📊 Wikipedia Web Traffic Time Series Forecasting")
    print("=" * 60)
    print("Models to be trained:")
    print("1. K-Nearest Neighbors (KNN) - Baseline")
    print("2. Long Short-Term Memory (LSTM) - Deep Learning")
    print("3. Seq2Seq Causal CNN - Advanced CNN")
    print("=" * 60)
    
    # Initialize and run pipeline
    pipeline = WikiTrafficForecastingPipeline(
        data_path=DATA_PATH,
        sample_size=SAMPLE_SIZE,
        results_dir=RESULTS_DIR
    )
    
    # Run complete pipeline
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n✅ All models trained and evaluated successfully!")
        print(f"📁 Check the '{RESULTS_DIR}' directory for detailed results and plots.")
    else:
        print("\n❌ Pipeline execution failed. Check the error messages above.")


if __name__ == "__main__":
    main()
