from fastapi import APIRouter, HTTPException, Query
from typing import List
import logging

from app.api.schemas.request import (
    SpamPredictionInput,
    SpamPredictionOutput,
    BatchPredictionInput,
    BatchPredictionOutput,
    ModelsListOutput,
    ModelInfo,
    AllModelsPredictionInput,
    AllModelsPredictionOutput,
    ModelPredictionResult,
    ModelType
)
from app.api.models.spam_model import get_spam_detection_model

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/predict", response_model=SpamPredictionOutput, tags=["Prediction"])
async def predict_spam(input_data: SpamPredictionInput):
    """
    Predict whether a message is spam using the specified model.
    
    - text: The message text to analyze (required)
    - model: The model to use for prediction (optional, defaults to logistic_regression)
    
    Returns classification results with confidence scores.
    """
    try:
        # Get the model instance
        spam_model = get_spam_detection_model()
        
        # Make prediction
        result = spam_model.predict(
            text=input_data.text,
            model_name=input_data.model.value
        )
        
        return SpamPredictionOutput(result)
        
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        logger.error(f"Runtime error: {re}")
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        logger.error(f"Unexpected error in predict_spam: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during prediction")


@router.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Prediction"])
async def predict_spam_batch(input_data: BatchPredictionInput):
    """
    Predict whether multiple messages are spam using the specified model.
    
    - messages: List of message texts to analyze (1-100 messages)
    - model: The model to use for predictions (optional, defaults to logistic_regression)
    
    Returns classification results for all messages with summary statistics.
    """
    try:
        # Get the model instance
        spam_model = get_spam_detection_model()
        
        # Make batch predictions
        results = spam_model.predict_batch(
            texts=input_data.messages,
            model_name=input_data.model.value
        )
        
        # Convert results to SpamPredictionOutput objects
        predictions = [SpamPredictionOutput(result) for result in results]
        
        # Calculate summary statistics
        spam_count = sum(1 for pred in predictions if pred.is_spam)
        ham_count = len(predictions) - spam_count
        
        return BatchPredictionOutput(
            predictions=predictions,
            total_messages=len(predictions),
            spam_count=spam_count,
            ham_count=ham_count,
            model_used=input_data.model.value
        )
        
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        logger.error(f"Runtime error: {re}")
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        logger.error(f"Unexpected error in predict_spam_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during batch prediction")


@router.post("/predict/all-models", response_model=AllModelsPredictionOutput, tags=["Prediction"])
async def predict_with_all_models(input_data: AllModelsPredictionInput):
    """
    Predict whether a message is spam using ALL available models.
    
    - text: The message text to analyze with all models
    
    Returns predictions from all models with consensus information.
    """
    try:
        # Get the model instance
        spam_model = get_spam_detection_model()
        
        # Get predictions from all models
        predictions = spam_model.predict_all_models(text=input_data.text)
        
        # Convert to ModelPredictionResult objects
        prediction_results = [ModelPredictionResult(pred) for pred in predictions]
        
        # Calculate consensus
        spam_votes = sum(1 for pred in prediction_results if pred.is_spam)
        total_models = len(prediction_results)
        
        if spam_votes > total_models / 2:
            consensus = "spam"
            consensus_confidence = (spam_votes / total_models) * 100
        else:
            consensus = "ham"
            consensus_confidence = ((total_models - spam_votes) / total_models) * 100
        
        return AllModelsPredictionOutput(
            text=input_data.text,
            predictions=prediction_results,
            consensus=consensus,
            consensus_confidence=round(consensus_confidence, 2)
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in predict_with_all_models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during multi-model prediction")


@router.get("/models", response_model=ModelsListOutput, tags=["Models"])
async def get_available_models():
    """
    Get information about all available spam detection models.
    
    Returns a list of models with their performance metrics and metadata.
    """
    try:
        # Get the model instance
        spam_model = get_spam_detection_model()
        
        # Get information about all models
        models_info = spam_model.get_all_models_info()
        
        # Convert to ModelInfo objects
        model_list = []
        for info in models_info:
            model_list.append(ModelInfo(
                name=info['name'],
                display_name=info['display_name'],
                accuracy=info.get('accuracy', 0.0),
                precision=info.get('precision', 0.0),
                recall=info.get('recall', 0.0),
                f1_score=info.get('f1_score', 0.0),
                description=info.get('description', '')
            ))
        
        return ModelsListOutput(
            models=model_list,
            default_model="logistic_regression",
            total_models=len(model_list)
        )
        
    except Exception as e:
        logger.error(f"Error getting models info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve models information")


@router.get("/models/{model_name}", response_model=ModelInfo, tags=["Models"])
async def get_model_info(model_name: ModelType):
    """
    Get detailed information about a specific model.
    
    - model_name: The name of the model to get information about
    
    Returns detailed model information including performance metrics.
    """
    try:
        # Get the model instance
        spam_model = get_spam_detection_model()
        
        # Get model information
        info = spam_model.get_model_info(model_name.value)
        
        if info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name.value}' not found"
            )
        
        return ModelInfo(
            name=info['name'],
            display_name=info['display_name'],
            accuracy=info.get('accuracy', 0.0),
            precision=info.get('precision', 0.0),
            recall=info.get('recall', 0.0),
            f1_score=info.get('f1_score', 0.0),
            description=info.get('description', '')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")


@router.post("/reload-models", tags=["Admin"])
async def reload_models_endpoint():
    """
    Reload all machine learning models (admin endpoint).
    
    Useful when models have been updated or retrained.
    Requires proper authentication in production.
    """
    try:
        from app.api.models.spam_model import reload_models
        
        spam_model = reload_models()
        models_info = spam_model.get_all_models_info()
        
        return {
            "status": "success",
            "message": "Models reloaded successfully",
            "models_loaded": len(models_info),
            "models": [info['name'] for info in models_info]
        }
        
    except Exception as e:
        logger.error(f"Error reloading models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reload models: {str(e)}")


@router.get("/test-messages", tags=["Testing"])
async def get_test_messages():
    """
    Get a set of test messages for trying out the spam detection.
    
    Returns example spam and ham messages for testing purposes.
    """
    return {
        "test_messages": [
            {
                "id": 1,
                "label": "spam",
                "text": "WINNER!! You have won a £1000 cash prize! Call 09061701461 to claim your prize now!"
            },
            {
                "id": 2,
                "label": "spam",
                "text": "Congratulations! You've been selected for a FREE iPhone 15. Click here: bit.ly/free-phone"
            },
            {
                "id": 3,
                "label": "spam",
                "text": "URGENT: Your bank account has been suspended. Verify your details immediately at secure-bank-login.com"
            },
            {
                "id": 4,
                "label": "ham",
                "text": "Hey, are we still meeting for lunch tomorrow at 1pm? Let me know if you need to reschedule."
            },
            {
                "id": 5,
                "label": "ham",
                "text": "Thanks for helping me move last weekend! I really appreciate it. Let's grab coffee soon."
            },
            {
                "id": 6,
                "label": "spam",
                "text": "Dear Customer, Your package delivery failed. Pay £2.99 redelivery fee at: parcel-redelivery24.com"
            },
            {
                "id": 7,
                "label": "ham",
                "text": "Don't forget we have the team meeting at 3 PM today. I'll send the agenda shortly."
            },
            {
                "id": 8,
                "label": "spam",
                "text": "MAKE $5000 PER WEEK FROM HOME! No experience needed! Limited spots available! Click now!"
            }
        ]
    }
