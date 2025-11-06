from pydantic import BaseModel, validator, Field
from typing import Optional, List, Dict
from enum import Enum


class ModelType(str, Enum):
    """Available spam detection models"""
    LOGISTIC_REGRESSION = "logistic_regression"
    NAIVE_BAYES = "naive_bayes"
    RANDOM_FOREST = "random_forest"
    SVM = "svm"


class SpamPredictionInput(BaseModel):
    """Input schema for spam prediction request"""
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The message text to analyze for spam",
        example="Congratulations! You've won a $1000 prize. Click here to claim now!"
    )
    model: Optional[ModelType] = Field(
        default=ModelType.LOGISTIC_REGRESSION,
        description="The model to use for prediction"
    )
    
    @validator('text')
    def validate_text(cls, v):
        """Validate and clean the input text"""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or contain only whitespace")
        
        # Remove excessive whitespace
        v = ' '.join(v.split())
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "text": "URGENT: Your account will be closed. Click here immediately!",
                "model": "logistic_regression"
            }
        }


class SpamPredictionOutput(BaseModel):
    """Output schema for spam prediction response"""
    is_spam: bool = Field(..., description="Whether the message is classified as spam")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    spam_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of being spam")
    ham_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of being ham (not spam)")
    model_used: str = Field(..., description="The model that was used for prediction")
    label: str = Field(..., description="Classification label: 'spam' or 'ham'")
    
    class Config:
        schema_extra = {
            "example": {
                "is_spam": True,
                "confidence": 0.95,
                "spam_probability": 0.95,
                "ham_probability": 0.05,
                "model_used": "logistic_regression",
                "label": "spam"
            }
        }


class BatchPredictionInput(BaseModel):
    """Input schema for batch spam prediction"""
    messages: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of messages to analyze"
    )
    model: Optional[ModelType] = Field(
        default=ModelType.LOGISTIC_REGRESSION,
        description="The model to use for predictions"
    )
    
    @validator('messages')
    def validate_messages(cls, v):
        """Validate each message in the batch"""
        if not v:
            raise ValueError("Messages list cannot be empty")
        
        cleaned_messages = []
        for msg in v:
            if not msg or not msg.strip():
                raise ValueError("Messages cannot be empty or contain only whitespace")
            cleaned_messages.append(' '.join(msg.split()))
        
        return cleaned_messages


class BatchPredictionOutput(BaseModel):
    """Output schema for batch spam prediction response"""
    predictions: List[SpamPredictionOutput]
    total_messages: int
    spam_count: int
    ham_count: int
    model_used: str


class ModelInfo(BaseModel):
    """Information about a specific model"""
    name: str
    display_name: str
    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    description: Optional[str] = None


class ModelsListOutput(BaseModel):
    """Output schema for available models list"""
    models: List[ModelInfo]
    default_model: str
    total_models: int


class AllModelsPredictionInput(BaseModel):
    """Input schema for prediction using all models"""
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The message text to analyze with all models"
    )
    
    @validator('text')
    def validate_text(cls, v):
        """Validate and clean the input text"""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or contain only whitespace")
        return ' '.join(v.split())


class ModelPredictionResult(BaseModel):
    """Result from a single model"""
    model_name: str
    is_spam: bool
    confidence: float
    spam_probability: float
    ham_probability: float


class AllModelsPredictionOutput(BaseModel):
    """Output schema for predictions from all models"""
    text: str
    predictions: List[ModelPredictionResult]
    consensus: str = Field(..., description="Overall consensus: 'spam' or 'ham'")
    consensus_confidence: float = Field(..., description="Percentage of models agreeing")


class ErrorResponse(BaseModel):
    """Standard error response schema"""
    detail: str
    path: Optional[str] = None
    timestamp: Optional[str] = None
