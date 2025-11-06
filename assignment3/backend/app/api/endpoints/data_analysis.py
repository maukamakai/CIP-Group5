from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict
import logging
import pandas as pd
from pathlib import Path

router = APIRouter(tags=["Data Analysis"])
logger = logging.getLogger(__name__)

# Path to the data file
DATA_PATH = Path("./app/data/spam_data_cleaned.csv")


@router.get("/data/stats")
async def get_data_statistics():
    """
    Get statistics about the spam dataset.
    
    Returns overview of the dataset including:
    - Total messages
    - Spam vs Ham distribution
    - Average message length
    - Data quality metrics
    """
    try:
        if not DATA_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail="Dataset not found. Please ensure spam_data_cleaned.csv is in app/data/"
            )
        
        # Load data
        df = pd.read_csv(DATA_PATH)
        
        # Calculate statistics
        total_messages = len(df)
        spam_count = len(df[df['label'] == 'spam'])
        ham_count = len(df[df['label'] == 'ham'])
        
        # Calculate average lengths
        df['text_length'] = df['spam_text'].str.len()
        avg_spam_length = df[df['label'] == 'spam']['text_length'].mean()
        avg_ham_length = df[df['label'] == 'ham']['text_length'].mean()
        
        # Get column info
        columns_available = list(df.columns)
        
        stats = {
            "total_messages": total_messages,
            "spam_count": spam_count,
            "ham_count": ham_count,
            "spam_percentage": round((spam_count / total_messages) * 100, 2),
            "ham_percentage": round((ham_count / total_messages) * 100, 2),
            "average_spam_length": round(avg_spam_length, 2),
            "average_ham_length": round(avg_ham_length, 2),
            "columns_available": columns_available,
            "data_shape": {
                "rows": df.shape[0],
                "columns": df.shape[1]
            }
        }
        
        logger.info("Data statistics retrieved successfully")
        return stats
        
    except Exception as e:
        logger.error(f"Error getting data statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.get("/data/samples")
async def get_data_samples(
    label: Optional[str] = Query(None, description="Filter by label: 'spam' or 'ham'"),
    limit: int = Query(10, ge=1, le=100, description="Number of samples to return")
):
    """
    Get sample messages from the dataset.
    
    - **label**: Optional filter ('spam' or 'ham')
    - **limit**: Number of samples (1-100)
    
    Returns random sample messages with their labels.
    """
    try:
        if not DATA_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail="Dataset not found"
            )
        
        # Load data
        df = pd.read_csv(DATA_PATH)
        
        # Filter by label if specified
        if label:
            if label.lower() not in ['spam', 'ham']:
                raise HTTPException(
                    status_code=400,
                    detail="Label must be 'spam' or 'ham'"
                )
            df = df[df['label'] == label.lower()]
        
        # Get random samples
        samples = df.sample(n=min(limit, len(df)))[['label', 'spam_text']]
        
        result = {
            "total_available": len(df),
            "returned": len(samples),
            "samples": [
                {
                    "label": row['label'],
                    "text": row['spam_text']
                }
                for _, row in samples.iterrows()
            ]
        }
        
        logger.info(f"Retrieved {len(samples)} sample messages")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting samples: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get samples: {str(e)}")


@router.get("/data/search")
async def search_messages(
    query: str = Query(..., min_length=1, description="Search term"),
    label: Optional[str] = Query(None, description="Filter by label: 'spam' or 'ham'"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results to return")
):
    """
    Search for messages containing specific text.
    
    - **query**: Text to search for (case-insensitive)
    - **label**: Optional filter ('spam' or 'ham')
    - **limit**: Maximum results (1-100)
    
    Returns messages matching the search query.
    """
    try:
        if not DATA_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail="Dataset not found"
            )
        
        # Load data
        df = pd.read_csv(DATA_PATH)
        
        # Filter by label if specified
        if label:
            if label.lower() not in ['spam', 'ham']:
                raise HTTPException(
                    status_code=400,
                    detail="Label must be 'spam' or 'ham'"
                )
            df = df[df['label'] == label.lower()]
        
        # Search for query in text (case-insensitive)
        mask = df['spam_text'].str.contains(query, case=False, na=False)
        results = df[mask][['label', 'spam_text']].head(limit)
        
        response = {
            "query": query,
            "total_matches": mask.sum(),
            "returned": len(results),
            "results": [
                {
                    "label": row['label'],
                    "text": row['spam_text']
                }
                for _, row in results.iterrows()
            ]
        }
        
        logger.info(f"Search for '{query}' found {mask.sum()} matches")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching messages: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/data/distribution")
async def get_label_distribution():
    """
    Get detailed distribution analysis of spam vs ham messages.
    
    Returns:
    - Count and percentage of each label
    - Length statistics for each category
    - Top words analysis
    """
    try:
        if not DATA_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail="Dataset not found"
            )
        
        # Load data
        df = pd.read_csv(DATA_PATH)
        
        # Calculate distributions
        label_counts = df['label'].value_counts().to_dict()
        total = len(df)
        
        # Length statistics
        df['length'] = df['spam_text'].str.len()
        
        spam_df = df[df['label'] == 'spam']
        ham_df = df[df['label'] == 'ham']
        
        distribution = {
            "total_messages": total,
            "spam": {
                "count": label_counts.get('spam', 0),
                "percentage": round((label_counts.get('spam', 0) / total) * 100, 2),
                "avg_length": round(spam_df['length'].mean(), 2),
                "min_length": int(spam_df['length'].min()),
                "max_length": int(spam_df['length'].max()),
                "median_length": round(spam_df['length'].median(), 2)
            },
            "ham": {
                "count": label_counts.get('ham', 0),
                "percentage": round((label_counts.get('ham', 0) / total) * 100, 2),
                "avg_length": round(ham_df['length'].mean(), 2),
                "min_length": int(ham_df['length'].min()),
                "max_length": int(ham_df['length'].max()),
                "median_length": round(ham_df['length'].median(), 2)
            }
        }
        
        logger.info("Distribution analysis completed")
        return distribution
        
    except Exception as e:
        logger.error(f"Error getting distribution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get distribution: {str(e)}")


@router.get("/data/random-message")
async def get_random_message(label: Optional[str] = Query(None, description="Filter by 'spam' or 'ham'")):
    """
    Get a single random message from the dataset.
    
    - **label**: Optional filter ('spam' or 'ham')
    
    Useful for quick testing of the prediction endpoints.
    """
    try:
        if not DATA_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail="Dataset not found"
            )
        
        # Load data
        df = pd.read_csv(DATA_PATH)
        
        # Filter by label if specified
        if label:
            if label.lower() not in ['spam', 'ham']:
                raise HTTPException(
                    status_code=400,
                    detail="Label must be 'spam' or 'ham'"
                )
            df = df[df['label'] == label.lower()]
        
        if len(df) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No messages found for label: {label}"
            )
        
        # Get random message
        random_msg = df.sample(n=1).iloc[0]
        
        result = {
            "label": random_msg['label'],
            "text": random_msg['spam_text'],
            "is_spam": random_msg['label'] == 'spam',
            "message": "Use this text to test the /predict endpoint!"
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting random message: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get random message: {str(e)}")


@router.get("/data/export-samples")
async def export_sample_data(
    count: int = Query(100, ge=10, le=1000, description="Number of samples to export")
):
    """
    Export a sample of the dataset for analysis.
    
    - **count**: Number of samples (10-1000)
    
    Returns balanced sample of spam and ham messages.
    """
    try:
        if not DATA_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail="Dataset not found"
            )
        
        # Load data
        df = pd.read_csv(DATA_PATH)
        
        # Get balanced sample (50% spam, 50% ham)
        spam_sample = df[df['label'] == 'spam'].sample(n=min(count // 2, len(df[df['label'] == 'spam'])))
        ham_sample = df[df['label'] == 'ham'].sample(n=min(count // 2, len(df[df['label'] == 'ham'])))
        
        # Combine and shuffle
        sample = pd.concat([spam_sample, ham_sample]).sample(frac=1)
        
        # Format for export
        export_data = [
            {
                "label": row['label'],
                "text": row['spam_text'],
                "is_spam": row['label'] == 'spam'
            }
            for _, row in sample.iterrows()
        ]
        
        result = {
            "total_exported": len(export_data),
            "spam_count": len(spam_sample),
            "ham_count": len(ham_sample),
            "data": export_data
        }
        
        logger.info(f"Exported {len(export_data)} sample messages")
        return result
        
    except Exception as e:
        logger.error(f"Error exporting samples: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to export samples: {str(e)}")
