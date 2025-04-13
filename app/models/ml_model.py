import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import optuna
from optuna.integration import XGBoostPruningCallback
import joblib
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, List, Optional, Union
import os
import json
import time
from datetime import datetime
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaintenanceModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.best_params = None
        self.feature_importances = None
        self.training_history = []
        self.model_directory = os.path.dirname(model_path)
        self.metadata_path = os.path.join(self.model_directory, 'model_metadata.json')
        
    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Objective function for Optuna optimization."""
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
            'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42
        }
        
        # K-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
            dval = xgb.DMatrix(X_val_fold, label=y_val_fold)
            
            # Train model
            model = xgb.train(
                param, 
                dtrain, 
                num_boost_round=100,
                evals=[(dval, 'validation')],
                early_stopping_rounds=20,
                verbose_eval=False
            )
            
            # Predict and evaluate
            preds = model.predict(dval)
            pred_labels = np.rint(preds)
            fold_score = f1_score(y_val_fold, pred_labels)
            cv_scores.append(fold_score)
        
        # Return the mean of cross-validation scores
        return np.mean(cv_scores)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, n_trials: int = 50) -> None:
        """Train the model with Optuna hyperparameter optimization and cross-validation."""
        try:
            start_time = time.time()
            
            # Create the model directory if it doesn't exist
            os.makedirs(self.model_directory, exist_ok=True)
            
            # Create a study for hyperparameter optimization
            study = optuna.create_study(direction="maximize", study_name="xgboost_optimization")
            study.optimize(lambda trial: self._objective(trial, X_train, y_train), n_trials=n_trials)
            
            # Train final model with the best parameters
            self.best_params = study.best_params
            self.best_params['objective'] = 'binary:logistic'
            self.best_params['eval_metric'] = 'logloss'
            
            # Train the final model with best parameters
            dtrain = xgb.DMatrix(X_train, label=y_train)
            final_model = xgb.train(self.best_params, dtrain, num_boost_round=100)
            
            # Store feature importances
            importance_type = 'gain'
            if hasattr(final_model, 'get_score'):
                self.feature_importances = final_model.get_score(importance_type=importance_type)
            
            # Create an ensemble model (optional)
            use_ensemble = False
            if use_ensemble:
                # Add other models to the ensemble
                xgb_model = xgb.XGBClassifier(**self.best_params)
                lr_model = LogisticRegression(max_iter=1000, random_state=42)
                
                # Train individual models
                xgb_model.fit(X_train, y_train)
                lr_model.fit(X_train, y_train)
                
                # Create voting classifier
                self.model = VotingClassifier(
                    estimators=[
                        ('xgb', xgb_model),
                        ('lr', lr_model)
                    ],
                    voting='soft'
                )
                self.model.fit(X_train, y_train)
            else:
                # Use the XGBoost model directly
                self.model = xgb.XGBClassifier(**self.best_params)
                self.model.fit(X_train, y_train)
            
            # Save model to disk
            joblib.dump(self.model, self.model_path)
            
            # Record training metadata
            training_duration = time.time() - start_time
            training_record = {
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'duration_seconds': training_duration,
                'best_params': self.best_params,
                'feature_importances': self.feature_importances,
                'dataset_size': len(X_train)
            }
            self.training_history.append(training_record)
            
            # Save metadata
            self._save_metadata()
            
            logger.info(f"Model trained successfully with best parameters: {self.best_params}")
            logger.info(f"Training completed in {training_duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def _save_metadata(self) -> None:
        """Save model metadata to disk."""
        try:
            metadata = {
                'training_history': self.training_history,
                'best_params': self.best_params,
                'feature_importances': self.feature_importances,
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Model metadata saved to {self.metadata_path}")
        except Exception as e:
            logger.error(f"Error saving model metadata: {str(e)}")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model performance with comprehensive metrics."""
        try:
            if self.model is None:
                raise ValueError("Model not trained yet")
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # For ROC AUC, we need probability scores
            if hasattr(self.model, 'predict_proba'):
                y_prob = self.model.predict_proba(X_test)[:, 1]
            else:
                y_prob = y_pred  # Fallback if probabilities aren't available
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_prob)
            }
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Calculate cross-validation score
            cv_scores = cross_val_score(self.model, X_test, y_test, cv=5, scoring='f1')
            metrics['cv_f1_mean'] = np.mean(cv_scores)
            metrics['cv_f1_std'] = np.std(cv_scores)
            
            logger.info(f"Model evaluation completed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions and return probabilities with improved error handling."""
        try:
            # Input validation
            if not isinstance(X, np.ndarray):
                raise ValueError("Input X must be a numpy array")
            
            if len(X.shape) != 2:
                raise ValueError(f"Expected 2D array, got {len(X.shape)}D array")

            # Load model if not loaded
            if self.model is None:
                try:
                    self.load_model()
                except Exception as e:
                    raise ValueError(f"Failed to load model: {str(e)}")
                    
                if self.model is None:
                    raise ValueError("Model not trained yet and could not be loaded")
            
            try:
                # Make predictions
                predictions = self.model.predict(X)
                
                # Get probability scores
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(X)
                else:
                    # If model doesn't support probability estimates, use predictions
                    probabilities = np.zeros((X.shape[0], 2))
                    probabilities[:, 1] = predictions
                    probabilities[:, 0] = 1 - predictions
                    logger.warning("Model does not support probability estimates, using binary predictions")
                
                # Validate outputs
                if not isinstance(predictions, np.ndarray):
                    predictions = np.array(predictions)
                if not isinstance(probabilities, np.ndarray):
                    probabilities = np.array(probabilities)
                
                # Ensure binary classification predictions
                if not np.all(np.isin(predictions, [0, 1])):
                    logger.warning("Non-binary predictions detected, rounding to nearest integer")
                    predictions = np.rint(predictions).astype(int)
                
                # Ensure valid probabilities
                probabilities = np.clip(probabilities, 0, 1)
                
                return predictions, probabilities
                
            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
                raise ValueError(f"Failed to make predictions: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error in predict method: {str(e)}")
            raise
    
    def load_model(self) -> None:
        """Load a trained model from disk."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                
                # Load metadata if available
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self.best_params = metadata.get('best_params')
                        self.feature_importances = metadata.get('feature_importances')
                        self.training_history = metadata.get('training_history', [])
                
                logger.info("Model loaded successfully")
            else:
                logger.warning("No trained model found at specified path")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.feature_importances:
            return self.feature_importances
        elif self.model and hasattr(self.model, 'feature_importances_'):
            return {f"feature_{i}": importance for i, importance in enumerate(self.model.feature_importances_)}
        else:
            return {} 