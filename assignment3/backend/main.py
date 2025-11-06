from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
import time
from app.api.endpoints import spam_detection, data_analysis

# Initialize FastAPI app
app = FastAPI(title="Spam and Ham Detection API")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Middleware to measure response time
@app.middleware("http")
async def measure_response_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    end_time = time.time()
    response_time = end_time - start_time
    
    # Log performance for key endpoints
    if request.url.path in ["/predict", "/models", "/health"]:
        logger.info(
            f"{request.method} {request.url.path} completed in {response_time:.4f} seconds"
        )
    
    # Add response time header
    response.headers["X-Response-Time"] = f"{response_time:.4f}"
    
    return response

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this based on your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Included API routers
app.include_router(spam_detection.router)
app.include_router(data_analysis.router)

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Spam Detection API",
        "version": "1.0.0",
        "endpoints": {
            "prediction": {
                "predict": "/predict",
                "batch": "/predict/batch",
                "all_models": "/predict/all-models"
            },
            "models": {
                "list": "/models",
                "info": "/models/{model_name}"
            },
            "data_analysis": {
                "stats": "/data/stats",
                "samples": "/data/samples",
                "search": "/data/search",
                "distribution": "/data/distribution",
                "random": "/data/random-message"
            },
            "utility": {
                "health": "/health",
                "test_messages": "/test-messages",
                "docs": "/docs"
            }
        }
    }

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "spam-detection-api"
}

# Exception handler for HTTPException
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(
        f"Request to {request.url.path} failed with status {exc.status_code}: {exc.detail}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "path": str(request.url.path)},
    )

# Exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "A server error occurred.",
            "path": str(request.url.path)
        },
    )

# Exception handler for request validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": exc.body,
            "path": str(request.url.path)
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
