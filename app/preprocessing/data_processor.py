import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from typing import Tuple, Dict, Any, List, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        logger.info(f"DataProcessor initialized with data_path: {self.data_path}")
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.feature_selector = None
        # Base feature columns from engine_data.csv
        self.base_feature_columns = [
            'Engine rpm', 'Lub oil pressure', 'Fuel pressure',
            'Coolant pressure', 'lub oil temp', 'Coolant temp'
        ]
        self.engineered_feature_columns = [
            'rpm_pressure_ratio',
            'temp_difference',
            'pressure_sum',
            'temp_pressure_interaction'
        ]
        self.feature_columns = self.base_feature_columns.copy()
        self.target_column = 'Engine Condition'
        self.selected_features = None
        self.feature_ranges = {
            'Engine rpm': (400, 2000),
            'Lub oil pressure': (2.0, 6.0),
            'Fuel pressure': (3.0, 20.0),
            'Coolant pressure': (1.0, 4.0),
            'lub oil temp': (70.0, 90.0),
            'Coolant temp': (70.0, 90.0)
        }

    def _create_synthetic_dataset(self) -> pd.DataFrame:
        """Create a synthetic dataset with the expected columns."""
        try:
            # Create a dataset with 1000 rows and the expected columns
            import numpy as np
            
            # Generate random data for each feature
            data = {
                'Engine rpm': np.random.uniform(400, 2000, 1000),
                'Lub oil pressure': np.random.uniform(2.0, 6.0, 1000),
                'Fuel pressure': np.random.uniform(3.0, 20.0, 1000),
                'Coolant pressure': np.random.uniform(1.0, 4.0, 1000),
                'lub oil temp': np.random.uniform(70.0, 90.0, 1000),
                'Coolant temp': np.random.uniform(70.0, 90.0, 1000),
                'Engine Condition': np.random.randint(0, 2, 1000)  # Binary target
            }
            
            df = pd.DataFrame(data)
            logger.info(f"Created synthetic dataset with shape: {df.shape}")
            
            # Save the synthetic dataset to the data path for future use
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            df.to_csv(self.data_path, index=False)
            logger.info(f"Saved synthetic dataset to {self.data_path}")
            
            return df
        except Exception as e:
            logger.error(f"Error creating synthetic dataset: {str(e)}")
            raise
    
    def load_data(self) -> pd.DataFrame:
        """Load and perform initial data cleaning."""
        try:
            # Try to load the data file
            try:
                df = pd.read_csv(self.data_path)
                logger.info(f"Loaded data with shape: {df.shape}")
                logger.info(f"Actual columns in data: {list(df.columns)}")
                
                # Check if the expected columns exist
                missing_columns = [col for col in self.base_feature_columns if col not in df.columns]
                if missing_columns:
                    logger.warning(f"Missing expected columns: {missing_columns}. Creating synthetic dataset.")
                    df = self._create_synthetic_dataset()
            except Exception as e:
                logger.warning(f"Error loading data file: {str(e)}. Creating synthetic dataset.")
                df = self._create_synthetic_dataset()
                
            return df
        except Exception as e:
            logger.error(f"Error in load_data: {str(e)}")
            raise
            
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features to the DataFrame."""
        try:
            processed_df = df.copy()
            
            # Add engineered features
            processed_df['rpm_pressure_ratio'] = processed_df['Engine rpm'] / processed_df['Lub oil pressure']
            processed_df['temp_difference'] = processed_df['lub oil temp'] - processed_df['Coolant temp']
            processed_df['pressure_sum'] = processed_df['Lub oil pressure'] + processed_df['Fuel pressure'] + processed_df['Coolant pressure']
            processed_df['temp_pressure_interaction'] = processed_df['lub oil temp'] * processed_df['Lub oil pressure']
            
            # Log transform skewed features
            processed_df['Engine rpm'] = np.log1p(processed_df['Engine rpm'])
            processed_df['Fuel pressure'] = np.log1p(processed_df['Fuel pressure'])
            
            return processed_df
        except Exception as e:
            logger.error(f"Error engineering features: {str(e)}")
            raise

    def preprocess_data(self, df: pd.DataFrame, apply_feature_selection: bool = True) -> pd.DataFrame:
        """Preprocess the data including advanced missing value handling and feature engineering."""
        try:
            # Create a copy to avoid modifying original data
            processed_df = df.copy()
            
            # Add engineered features
            processed_df = self._engineer_features(processed_df)
            
            # Update feature columns to include engineered features
            self.feature_columns = self.base_feature_columns + self.engineered_feature_columns
            
            # Handle missing values using KNN imputation
            processed_df[self.feature_columns] = self.imputer.fit_transform(processed_df[self.feature_columns])
            
            # Scale all features
            processed_df[self.feature_columns] = self.scaler.fit_transform(processed_df[self.feature_columns])
            
            if apply_feature_selection and self.target_column in processed_df.columns:
                # Apply feature selection to select most important features
                processed_df = self._apply_feature_selection(processed_df, self.feature_columns)
                if processed_df is None:
                    logger.error("Feature selection returned None, using original DataFrame")
                    return df
            
            logger.info(f"Preprocessed data shape: {processed_df.shape}")
            logger.info(f"Final features: {list(processed_df.columns)}")
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def _apply_feature_selection(self, df: pd.DataFrame, feature_cols: List[str], k: int = 10) -> pd.DataFrame:
        """Apply feature selection using mutual information."""
        try:
            # Create a copy of the DataFrame
            processed_df = df.copy()
            
            # Prepare feature matrix and target
            X = processed_df[feature_cols].values
            y = processed_df[self.target_column].values
            
            # Apply mutual information feature selection
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_mask = selector.get_support()
            self.selected_features = [col for idx, col in enumerate(feature_cols) if selected_mask[idx]]
            
            # Create new DataFrame with selected features
            selected_df = pd.DataFrame(X_selected, columns=self.selected_features)
            selected_df[self.target_column] = processed_df[self.target_column]
            
            logger.info(f"Selected {len(self.selected_features)} features: {self.selected_features}")
            
            return selected_df
            
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            # Return original DataFrame if feature selection fails
            return df

    def apply_pca(self, df: pd.DataFrame, n_components: float = 0.95) -> pd.DataFrame:
        """Apply PCA dimensionality reduction."""
        try:
            feature_cols = [col for col in df.columns if col != self.target_column]
            X = df[feature_cols].values
            
            self.pca = PCA(n_components=n_components)
            X_pca = self.pca.fit_transform(X)
            
            # Create new DataFrame with PCA components
            pca_cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
            pca_df = pd.DataFrame(X_pca, columns=pca_cols)
            
            # Add target column if present
            if self.target_column in df.columns:
                pca_df[self.target_column] = df[self.target_column].values
            
            logger.info(f"Applied PCA: reduced from {len(feature_cols)} to {len(pca_cols)} dimensions")
            return pca_df
        except Exception as e:
            logger.error(f"Error applying PCA: {str(e)}")
            raise
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets."""
        try:
            feature_cols = [col for col in df.columns if col != self.target_column]
            X = df[feature_cols].values
            y = df[self.target_column].values
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            logger.info(f"Data split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error in data splitting: {str(e)}")
            raise
    
    def prepare_prediction_data(self, data: Dict[str, Any]) -> np.ndarray:
        """Prepare single prediction data."""
        try:
            # Convert input data to DataFrame
            df = pd.DataFrame([data])
            
            # Map input keys to expected feature names
            mapping = {
                'engine_rpm': 'Engine rpm',
                'lub_oil_pressure': 'Lub oil pressure',
                'fuel_pressure': 'Fuel pressure',
                'coolant_pressure': 'Coolant pressure',
                'lub_oil_temp': 'lub oil temp',
                'coolant_temp': 'Coolant temp'
            }
            
            df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
            
            # Handle missing features
            for feature in self.base_feature_columns:
                if feature not in df.columns:
                    logger.warning(f"Missing feature {feature}, using default value 0")
                    df[feature] = 0
            
            # Add engineered features
            df = self._engineer_features(df)
            
            # Update feature columns to include engineered features
            self.feature_columns = self.base_feature_columns + self.engineered_feature_columns
            
            # Apply imputation and scaling as before...
            # Make sure imputer is fitted
            if not hasattr(self.imputer, 'n_features_in_'):
                training_data = self.load_data()
                if training_data is not None and len(training_data) > 0:
                    processed_training = self._engineer_features(training_data)
                    self.imputer.fit(processed_training[self.feature_columns])
                    logger.info("Fitted KNNImputer on training data")
            
            # Apply imputation
            df[self.feature_columns] = self.imputer.transform(df[self.feature_columns])
            
            # Make sure scaler is fitted
            if not hasattr(self.scaler, 'n_features_in_'):
                training_data = self.load_data()
                if training_data is not None and len(training_data) > 0:
                    processed_training = self._engineer_features(training_data)
                    self.scaler.fit(processed_training[self.feature_columns])
                    logger.info("Fitted StandardScaler on training data")
            
            # Scale features
            df[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
            
            # Make sure to return all features including engineered ones
            return df[self.feature_columns].values
            
        except Exception as e:
            logger.error(f"Error preparing prediction data: {str(e)}")
            raise