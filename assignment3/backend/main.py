from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
import time
from app.api.endpoints import regression

# Initialize FastAPI app
app = FastAPI(title = "Spam and Ham Detection and Classifier API")
