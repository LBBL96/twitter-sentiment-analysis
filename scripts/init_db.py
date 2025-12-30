import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.database import init_db, engine
from src.database.models import Base
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_database():
    logger.info("Initializing database...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully!")
        logger.info("Tables created: posts, reddit_posts, training_data, model_metrics")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


if __name__ == "__main__":
    initialize_database()
