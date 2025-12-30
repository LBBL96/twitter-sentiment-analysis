from src.tasks.celery_app import celery_app
from src.model_training.retraining_pipeline import DataQualityMonitor
from src.database.database import SessionLocal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@celery_app.task
def monitor_data_quality():
    logger.info("Monitoring data quality...")
    
    db = SessionLocal()
    try:
        monitor = DataQualityMonitor(db)
        
        drift_info = monitor.check_data_drift(window_hours=24)
        performance_info = monitor.check_model_performance(window_hours=24)
        
        if drift_info['has_drift']:
            logger.warning(f"Data drift detected: {drift_info}")
        
        if performance_info.get('status') == 'degraded':
            logger.warning(f"Model performance degraded: {performance_info}")
        
        return {
            'drift': drift_info,
            'performance': performance_info
        }
    
    finally:
        db.close()


@celery_app.task
def log_metrics_summary():
    db = SessionLocal()
    try:
        from src.database.models import RedditPost
        from datetime import datetime, timedelta
        
        cutoff = datetime.utcnow() - timedelta(hours=1)
        recent_predictions = db.query(RedditPost).filter(
            RedditPost.created_at >= cutoff
        ).count()
        
        logger.info(f"Predictions in last hour: {recent_predictions}")
        return {'predictions_last_hour': recent_predictions}
    
    finally:
        db.close()
