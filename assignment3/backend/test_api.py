#!/usr/bin/env python3
"""
Test script for Spam Detection API
Tests all major endpoints to verify functionality
"""

import requests
import json
import sys
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

# ANSI color codes for pretty output
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def print_section(title: str):
    """Print a section header"""
    print(f"\n{BLUE}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{RESET}\n")


def print_success(message: str):
    """Print success message"""
    print(f"{GREEN}{message}{RESET}")


def print_error(message: str):
    """Print error message"""
    print(f"{RED}{message}{RESET}")


def print_info(message: str):
    """Print info message"""
    print(f"{YELLOW}{message}{RESET}")


def test_health_check():
    """Test the health check endpoint"""
    print_section("Testing Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print_success("Health check passed")
            print(f"Response: {response.json()}")
            return True
        else:
            print_error(f"Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Health check failed: {str(e)}")
        return False


def test_get_models():
    """Test getting available models"""
    print_section("Testing Get Models")
    try:
        response = requests.get(f"{BASE_URL}/models")
        if response.status_code == 200:
            data = response.json()
            print_success(f"Found {data['total_models']} models")
            for model in data['models']:
                print(f"{model['display_name']}")
                print(f"Accuracy: {model['accuracy']:.3f} | "
                      f"Precision: {model['precision']:.3f} | "
                      f"F1: {model['f1_score']:.3f}")
            return True
        else:
            print_error(f"Failed to get models: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Failed to get models: {str(e)}")
        return False


def test_single_prediction(text: str, expected_label: str):
    """Test single message prediction"""
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={
                "text": text,
                "model": "logistic_regression"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            actual_label = data['label']
            confidence = data['confidence']
            
            if actual_label == expected_label:
                print_success(f"Correct prediction: {actual_label} (confidence: {confidence:.2%})")
            else:
                print_info(f"Predicted: {actual_label} | Expected: {expected_label} (confidence: {confidence:.2%})")
            
            print(f"Text: {text[:60]}...")
            return True
        else:
            print_error(f"Prediction failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Prediction failed: {str(e)}")
        return False


def test_predictions():
    """Test predictions with various messages"""
    print_section("Testing Single Predictions")
    
    test_cases = [
        ("WINNER!! You have won a £1000 cash prize! Call now!", "spam"),
        ("Hey, are we still meeting for lunch tomorrow?", "ham"),
        ("URGENT: Your bank account has been suspended. Click here!", "spam"),
        ("Thanks for helping me move last weekend!", "ham"),
        ("FREE iPhone 15! Click here to claim your prize NOW!!!", "spam"),
    ]
    
    success_count = 0
    for text, expected in test_cases:
        if test_single_prediction(text, expected):
            success_count += 1
    
    print(f"\n{success_count}/{len(test_cases)} predictions completed")
    return success_count == len(test_cases)


def test_batch_prediction():
    """Test batch prediction"""
    print_section("Testing Batch Prediction")
    
    messages = [
        "Meeting at 3pm today",
        "WIN BIG MONEY NOW!!!",
        "Can you pick up milk?",
        "URGENT: Account suspended!",
    ]
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json={
                "messages": messages,
                "model": "logistic_regression"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Batch prediction completed")
            print(f"Total messages: {data['total_messages']}")
            print(f"Spam count: {data['spam_count']}")
            print(f"Ham count: {data['ham_count']}")
            
            for i, pred in enumerate(data['predictions']):
                label_color = RED if pred['is_spam'] else GREEN
                print(f"   {i+1}. {label_color}{pred['label'].upper()}{RESET} "
                      f"({pred['confidence']:.2%}) - {messages[i][:40]}...")
            
            return True
        else:
            print_error(f"Batch prediction failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Batch prediction failed: {str(e)}")
        return False


def test_all_models_prediction():
    """Test prediction with all models"""
    print_section("Testing All Models Prediction")
    
    text = "CONGRATULATIONS! You've won a FREE vacation! Click here NOW!"
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/all-models",
            json={"text": text}
        )
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"All models prediction completed")
            print(f"Consensus: {data['consensus'].upper()} ({data['consensus_confidence']:.1f}% agreement)")
            print(f"\nIndividual model predictions:")
            
            for pred in data['predictions']:
                label_color = RED if pred['is_spam'] else GREEN
                print(f"{pred['model_name']:<25} {label_color}{pred['is_spam']}{RESET} "
                      f"(confidence: {pred['confidence']:.2%})")
            
            return True
        else:
            print_error(f"All models prediction failed: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"All models prediction failed: {str(e)}")
        return False


def test_get_test_messages():
    """Test getting test messages"""
    print_section("Testing Get Test Messages")
    
    try:
        response = requests.get(f"{BASE_URL}/test-messages")
        if response.status_code == 200:
            data = response.json()
            print_success(f"Retrieved {len(data['test_messages'])} test messages")
            return True
        else:
            print_error(f"Failed to get test messages: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Failed to get test messages: {str(e)}")
        return False


def run_all_tests():
    """Run all tests"""
    print(f"{BLUE}")
    print("Spam Detection API - Testing")
    print(f"{RESET}")
    
    print_info(f"Testing API at: {BASE_URL}")
    print_info("Make sure the API server is running before running tests\n")
    
    results = []
    
    # Run all tests
    results.append(("Health Check", test_health_check()))
    results.append(("Get Models", test_get_models()))
    results.append(("Single Predictions", test_predictions()))
    results.append(("Batch Prediction", test_batch_prediction()))
    results.append(("All Models Prediction", test_all_models_prediction()))
    results.append(("Get Test Messages", test_get_test_messages()))
    
    # Print summary
    print_section("Test Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        if result:
            print_success(f"{test_name}")
        else:
            print_error(f"{test_name}")
    
    print(f"\n{BLUE}{'='*60}{RESET}")
    if passed == total:
        print(f"{GREEN}All tests passed! ({passed}/{total}){RESET}")
        return 0
    else:
        print(f"{YELLOW}{passed}/{total} tests passed{RESET}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = run_all_tests()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Tests interrupted by user{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{RED}Unexpected error: {str(e)}{RESET}")
        sys.exit(1)
