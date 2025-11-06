# Spam Detection API

A FastAPI application for spam detection using machine learning models from Assignment 2.

## Project Structure

```
backend_spam_detector/
├── app/
│   ├── api/
│   │   ├── endpoints/
│   │   │   ├── spam_detection.py    
│   │   │   └── data_analysis.py     
│   │   ├── models/
│   │   │   └── spam_model.py        
│   │   └── schemas/
│   │       └── request.py           
│   ├── ml_models/                   
│   └── data/
│       └── spam_data_cleaned.csv    
├── main.py                          
└── requirements.txt                 
```

## Installation

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the application
```bash
python main.py
```

The API will be available at http://127.0.0.1:8000

## API Documentation

FastAPI provides interactive API documentation at:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## Endpoints

### Prediction
- POST /predict - Single message prediction
- POST /predict/batch - Batch predictions
- POST /predict/all-models - Compare all models

### Models
- GET /models - List all available models
- GET /models/{name} - Get model details

### Data Analysis
- GET /data/stats - Dataset statistics
- GET /data/samples - Get sample messages
- GET /data/search - Search messages
- GET /data/distribution - Data distribution
- GET /data/random-message - Random message
- GET /data/export-samples - Export samples

### Utility
- GET /health - Health check
- GET /test-messages - Test messages
- POST /reload-models - Reload models

## Features

- 4 ML models (Logistic Regression, Naive Bayes, Random Forest, SVM)
- Real-time spam detection
- Batch processing support
- Comprehensive error handling
- Input validation with Pydantic
- Response time monitoring
