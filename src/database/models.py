from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean, JSON
from datetime import datetime
from src.database.database import Base


class Tweet(Base):
    __tablename__ = "tweets"
    
    id = Column(Integer, primary_key=True, index=True)
    tweet_id = Column(String, unique=True, index=True)
    text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    author_id = Column(String, nullable=True)
    lang = Column(String, nullable=True)
    preprocessed_text = Column(Text, nullable=True)
    sentiment_label = Column(String, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    model_version = Column(String, nullable=True)
    processed = Column(Boolean, default=False)
    metadata = Column(JSON, nullable=True)


class TrainingData(Base):
    __tablename__ = "training_data"
    
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text)
    label = Column(String)
    source = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    used_for_training = Column(Boolean, default=False)
    split = Column(String, nullable=True)


class RedditPost(Base):
    __tablename__ = "reddit_posts"
    
    id = Column(Integer, primary_key=True, index=True)
    post_id = Column(String, unique=True, index=True)
    text = Column(Text)
    title = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    author = Column(String, nullable=True)
    subreddit = Column(String, index=True)
    score = Column(Integer, nullable=True)
    num_comments = Column(Integer, nullable=True)
    url = Column(String, nullable=True)
    post_type = Column(String, nullable=True)
    preprocessed_text = Column(Text, nullable=True)
    sentiment_label = Column(String, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    model_version = Column(String, nullable=True)
    processed = Column(Boolean, default=False)
    metadata = Column(JSON, nullable=True)


class ModelMetrics(Base):
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String, index=True)
    model_path = Column(String)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    training_samples = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    mlflow_run_id = Column(String, nullable=True)
    wandb_run_id = Column(String, nullable=True)
    is_deployed = Column(Boolean, default=False)
    metadata = Column(JSON, nullable=True)
