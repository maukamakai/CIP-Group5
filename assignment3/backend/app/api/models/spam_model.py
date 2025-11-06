import os
import pickle
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class SpamDetectionModel:
    """
    Handler for spam detection models.
    Manages loading, prediction, and model information.
    """
    
    # Model display names and descriptions
    MODEL_INFO = {
        'logistic_regression': {
            'display_name': 'Logistic Regression',
            'description': 'Fast and efficient linear model for binary classification'
        },
        'naive_bayes': {
            'display_name': 'Naive Bayes',
            'description': 'Probabilistic classifier based on Bayes theorem'
        },
        'random_forest': {
            'display_name': 'Random Forest',
            'description': 'Ensemble learning method using multiple decision trees'
        },
        'svm': {
            'display_name': 'Support Vector Machine',
            'description': 'Powerful classifier that finds optimal decision boundaries'
        }
    }
    
    def __init__(self, models_dir: str = "./app/ml_models"):
        """
        Initialize the spam detection model handler.
        
        Args:
            models_dir: Directory containing the trained models
        """
        self.models_dir = Path(models_dir)
        self.vectorizer = None
        self.models = {}
        self.model_results = {}
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all trained models and the vectorizer."""
        try:
            # Load TF-IDF vectorizer
            vectorizer_path = self.models_dir / 'tfidf_vectorizer.pkl'
            if not vectorizer_path.exists():
                raise FileNotFoundError(
                    f"Vectorizer not found at {vectorizer_path}. "
                    "Please train models first using train_models.py"
                )
            
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            logger.info("Vectorizer loaded successfully")
            
            # Load all model files
            model_files = {
                'logistic_regression': 'logistic_regression_model.pkl',
                'naive_bayes': 'naive_bayes_model.pkl',
                'random_forest': 'random_forest_model.pkl',
                'svm': 'svm_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                model_path = self.models_dir / filename
                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    logger.info(f"Loaded model: {model_name}")
                else:
                    logger.warning(f"Model file not found: {filename}")
            
            # Load model results (performance metrics)
            results_path = self.models_dir / 'model_results.pkl'
            if results_path.exists():
                with open(results_path, 'rb') as f:
                    results = pickle.load(f)
                    # Convert display names to model keys
                    for display_name, metrics in results.items():
                        model_key = display_name.replace(' ', '_').lower()
                        if model_key in self.models:
                            self.model_results[model_key] = metrics
                logger.info("Model performance metrics loaded")
            else:
                logger.warning("Model results file not found")
            
            if not self.models:
                raise RuntimeError("No models were loaded successfully")
            
            logger.info(f"Successfully loaded {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load models: {str(e)}")
    
    def predict(self, text: str, model_name: str = 'logistic_regression') -> Dict:
        """
        Predict if a message is spam using the specified model.
        
        Args:
            text: The message text to classify
            model_name: Name of the model to use
            
        Returns:
            Dictionary containing prediction results
            
        Raises:
            ValueError: If model_name is not valid
            RuntimeError: If prediction fails
        """
        if model_name not in self.models:
            available_models = list(self.models.keys())
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {available_models}"
            )
        
        try:
            # Vectorize the input text
            text_vectorized = self.vectorizer.transform([text])
            
            # Get the model
            model = self.models[model_name]
            
            # Make prediction
            prediction = model.predict(text_vectorized)[0]
            probabilities = model.predict_proba(text_vectorized)[0]
            
            # Extract probabilities (assuming [ham, spam] order)
            ham_prob = float(probabilities[0])
            spam_prob = float(probabilities[1])
            
            # Determine confidence (max probability)
            confidence = max(ham_prob, spam_prob)
            
            result = {
                'is_spam': bool(prediction == 1),
                'confidence': round(confidence, 4),
                'spam_probability': round(spam_prob, 4),
                'ham_probability': round(ham_prob, 4),
                'model_used': model_name,
                'label': 'spam' if prediction == 1 else 'ham'
            }
            
            logger.info(
                f"Prediction made with {model_name}: "
                f"{'spam' if prediction == 1 else 'ham'} "
                f"(confidence: {confidence:.4f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error with {model_name}: {e}", exc_info=True)
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_all_models(self, text: str) -> List[Dict]:
        """
        Predict using all available models.
        
        Args:
            text: The message text to classify
            
        Returns:
            List of prediction results from all models
        """
        predictions = []
        
        for model_name in self.models.keys():
            try:
                result = self.predict(text, model_name)
                predictions.append({
                    'model_name': self.MODEL_INFO[model_name]['display_name'],
                    'is_spam': result['is_spam'],
                    'confidence': result['confidence'],
                    'spam_probability': result['spam_probability'],
                    'ham_probability': result['ham_probability']
                })
            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {e}")
                continue
        
        return predictions
    
    def predict_batch(self, texts: List[str], model_name: str = 'logistic_regression') -> List[Dict]:
        """
        Predict spam for multiple messages at once.
        
        Args:
            texts: List of message texts to classify
            model_name: Name of the model to use
            
        Returns:
            List of prediction results
        """
        if model_name not in self.models:
            available_models = list(self.models.keys())
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {available_models}"
            )
        
        results = []
        
        try:
            # Vectorize all texts at once (more efficient)
            texts_vectorized = self.vectorizer.transform(texts)
            
            # Get the model
            model = self.models[model_name]
            
            # Make predictions for all texts
            predictions = model.predict(texts_vectorized)
            probabilities = model.predict_proba(texts_vectorized)
            
            # Process results
            for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
                ham_prob = float(probs[0])
                spam_prob = float(probs[1])
                confidence = max(ham_prob, spam_prob)
                
                results.append({
                    'is_spam': bool(pred == 1),
                    'confidence': round(confidence, 4),
                    'spam_probability': round(spam_prob, 4),
                    'ham_probability': round(ham_prob, 4),
                    'model_used': model_name,
                    'label': 'spam' if pred == 1 else 'ham'
                })
            
            logger.info(f"Batch prediction completed: {len(texts)} messages")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}", exc_info=True)
            raise RuntimeError(f"Batch prediction failed: {str(e)}")
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary containing model information
        """
        if model_name not in self.models:
            return None
        
        info = {
            'name': model_name,
            'display_name': self.MODEL_INFO[model_name]['display_name'],
            'description': self.MODEL_INFO[model_name]['description'],
            'is_loaded': True
        }
        
        # Add performance metrics if available
        if model_name in self.model_results:
            metrics = self.model_results[model_name]
            info.update({
                'accuracy': round(metrics.get('accuracy', 0), 4),
                'precision': round(metrics.get('precision', 0), 4),
                'recall': round(metrics.get('recall', 0), 4),
                'f1_score': round(metrics.get('f1_score', 0), 4)
            })
        
        return info
    
    def get_all_models_info(self) -> List[Dict]:
        """
        Get information about all available models.
        
        Returns:
            List of dictionaries containing model information
        """
        models_info = []
        
        for model_name in self.models.keys():
            info = self.get_model_info(model_name)
            if info:
                models_info.append(info)
        
        return models_info
    
    def is_model_available(self, model_name: str) -> bool:
        """
        Check if a model is available.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is loaded and available
        """
        return model_name in self.models


# Global instance to be used by endpoints
_model_instance: Optional[SpamDetectionModel] = None


def get_spam_detection_model() -> SpamDetectionModel:
    """
    Get the global spam detection model instance.
    Creates it if it doesn't exist (singleton pattern).
    
    Returns:
        SpamDetectionModel instance
    """
    global _model_instance
    
    if _model_instance is None:
        _model_instance = SpamDetectionModel()
    
    return _model_instance


def reload_models():
    """Reload all models (useful for updates)."""
    global _model_instance
    _model_instance = None
    return get_spam_detection_model()
