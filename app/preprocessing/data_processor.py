import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from typing import Tuple, Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.feature_selector = None
        self.pca = None
        self.feature_columns = [
            'Engine rpm', 'Lub oil pressure', 'Fuel pressure',
            'Coolant pressure', 'lub oil temp', 'Coolant temp'
        ]
        self.target_column = 'Engine Condition'
        self.selected_features = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and perform initial data cleaning."""
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame, apply_feature_selection: bool = True) -> pd.DataFrame:
        """Preprocess the data including advanced missing value handling and feature engineering."""
        try:
            # Create a copy to avoid modifying original data
            processed_df = df.copy()
            
            # Handle missing values using KNN imputation
            processed_df[self.feature_columns] = self.imputer.fit_transform(processed_df[self.feature_columns])
            
            # Add engineered features - rolling averages for time series data if timestamp column exists
            if 'timestamp' in processed_df.columns:
                processed_df.sort_values('timestamp', inplace=True)
                for feature in self.feature_columns:
                    processed_df[f'{feature}_rolling_mean'] = processed_df[feature].rolling(window=5, min_periods=1).mean()
                    processed_df[f'{feature}_rolling_std'] = processed_df[feature].rolling(window=5, min_periods=1).std().fillna(0)
                
                # Add features for anomaly detection - z-score based
                for feature in self.feature_columns:
                    mean = processed_df[feature].mean()
                    std = processed_df[feature].std()
                    processed_df[f'{feature}_zscore'] = (processed_df[feature] - mean) / std
            
            # Scale numerical features
            feature_cols = [col for col in processed_df.columns if col != self.target_column and pd.api.types.is_numeric_dtype(processed_df[col])]
            processed_df[feature_cols] = self.scaler.fit_transform(processed_df[feature_cols])
            
            # Apply feature selection if requested
            if apply_feature_selection and self.target_column in processed_df.columns:
                self._apply_feature_selection(processed_df, feature_cols)
                processed_df = processed_df[[self.target_column] + self.selected_features]
            
            logger.info("Data preprocessing completed successfully")
            return processed_df
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def _apply_feature_selection(self, df: pd.DataFrame, feature_cols: List[str], k: int = 10) -> None:
        """Apply feature selection using mutual information."""
        try:
            X = df[feature_cols]
            y = df[self.target_column]
            
            # Use mutual information for feature selection
            k = min(k, len(feature_cols))
            self.feature_selector = SelectKBest(mutual_info_classif, k=k)
            self.feature_selector.fit(X, y)
            
            # Get selected feature indices
            selected_indices = self.feature_selector.get_support(indices=True)
            self.selected_features = [feature_cols[i] for i in selected_indices]
            
            logger.info(f"Selected {len(self.selected_features)} features: {self.selected_features}")
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            raise
    
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
            for feature in self.feature_columns:
                if feature not in df.columns:
                    df[feature] = 0  # Default value when missing
            
            # Apply KNN imputation if any missing values
            if df.isnull().any().any():
                df[self.feature_columns] = self.imputer.transform(df[self.feature_columns])
            
            # Add engineered features if used in training
            if self.selected_features:
                # Only process the selected features that were used in training
                available_features = [f for f in self.selected_features if f in self.feature_columns]
                
                # Scale only the necessary features
                scaled_data = self.scaler.transform(df[available_features])
                return scaled_data
            else:
                # Scale the original features
                scaled_data = self.scaler.transform(df[self.feature_columns])
                return scaled_data
                
        except Exception as e:
            logger.error(f"Error preparing prediction data: {str(e)}")
            raise 