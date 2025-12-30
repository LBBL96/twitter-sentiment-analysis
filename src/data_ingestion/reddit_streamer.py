import praw
import logging
from collections.abc import Callable
from datetime import datetime
from config import settings
from src.database.models import RedditPost
from src.database.database import SessionLocal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedditStreamListener:
    """
    Reddit API Stream Listener using PRAW (Python Reddit API Wrapper)
    
    Uses OAuth authentication with Reddit API (completely free).
    No credit card required - just create a Reddit app at:
    https://www.reddit.com/prefs/apps
    
    Features:
    - Real-time subreddit monitoring
    - Comment and submission streaming
    - Completely free with generous rate limits
    - Rich contextual data (longer posts than tweets)
    """
    
    def __init__(
        self,
        callback: Callable | None = None,
        save_to_db: bool = True,
    ):
        self.callback = callback
        self.save_to_db = save_to_db
        self.post_count = 0
        self.reddit = self._initialize_reddit()
        
    def _initialize_reddit(self) -> praw.Reddit:
        """Initialize Reddit API client with credentials."""
        return praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
        )
    
    def process_submission(self, submission):
        """Process a Reddit submission (post)."""
        try:
            post_data = {
                'post_id': submission.id,
                'text': f"{submission.title} {submission.selftext}".strip(),
                'title': submission.title,
                'created_at': datetime.fromtimestamp(submission.created_utc),
                'author': str(submission.author) if submission.author else '[deleted]',
                'subreddit': str(submission.subreddit),
                'score': submission.score,
                'num_comments': submission.num_comments,
                'url': submission.url,
                'post_type': 'submission',
            }
            
            self.post_count += 1
            logger.info(f"Received submission #{self.post_count}: r/{post_data['subreddit']} - {post_data['title'][:50]}")
            
            if self.save_to_db:
                self._save_to_database(post_data)
            
            if self.callback:
                self.callback(post_data)
                
        except Exception as e:
            logger.error(f"Error processing submission: {e}")
    
    def process_comment(self, comment):
        """Process a Reddit comment."""
        try:
            post_data = {
                'post_id': comment.id,
                'text': comment.body,
                'title': None,
                'created_at': datetime.fromtimestamp(comment.created_utc),
                'author': str(comment.author) if comment.author else '[deleted]',
                'subreddit': str(comment.subreddit),
                'score': comment.score,
                'num_comments': 0,
                'url': f"https://reddit.com{comment.permalink}",
                'post_type': 'comment',
            }
            
            self.post_count += 1
            logger.info(f"Received comment #{self.post_count}: r/{post_data['subreddit']} - {post_data['text'][:50]}")
            
            if self.save_to_db:
                self._save_to_database(post_data)
            
            if self.callback:
                self.callback(post_data)
                
        except Exception as e:
            logger.error(f"Error processing comment: {e}")
    
    def _save_to_database(self, post_data: dict):
        """Save Reddit post to database."""
        db = SessionLocal()
        try:
            reddit_post = RedditPost(**post_data)
            db.add(reddit_post)
            db.commit()
            logger.debug(f"Saved post {post_data['post_id']} to database")
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            db.rollback()
        finally:
            db.close()


class RedditStreamManager:
    """
    Reddit Stream Manager
    
    Manages Reddit API connections and streams data from specified subreddits.
    Completely free with no credit card required.
    
    Setup:
    1. Go to https://www.reddit.com/prefs/apps
    2. Click "Create App" or "Create Another App"
    3. Select "script" type
    4. Get your client_id and client_secret
    
    Rate limits:
    - 60 requests per minute (very generous)
    - No monthly limits or costs
    """
    
    def __init__(self):
        self.listener = None
        self.reddit = None
        
    def setup_listener(
        self,
        callback: Callable | None = None,
        save_to_db: bool = True
    ) -> RedditStreamListener:
        """Initialize Reddit stream listener."""
        self.listener = RedditStreamListener(
            callback=callback,
            save_to_db=save_to_db
        )
        self.reddit = self.listener.reddit
        return self.listener
    
    def stream_subreddit_submissions(
        self,
        subreddit_names: list[str],
        callback: Callable | None = None,
        save_to_db: bool = True,
        skip_existing: bool = True
    ):
        """
        Stream new submissions from specified subreddits.
        
        Args:
            subreddit_names: List of subreddit names (without 'r/')
            callback: Optional callback function for each post
            save_to_db: Whether to save posts to database
            skip_existing: Skip already processed submissions
        """
        if not self.listener:
            self.setup_listener(callback=callback, save_to_db=save_to_db)
        
        subreddit_str = '+'.join(subreddit_names)
        subreddit = self.reddit.subreddit(subreddit_str)
        
        logger.info(f"Starting Reddit submission stream for: r/{subreddit_str}")
        
        try:
            for submission in subreddit.stream.submissions(skip_existing=skip_existing):
                self.listener.process_submission(submission)
        except Exception as e:
            logger.error(f"Error in submission stream: {e}")
    
    def stream_subreddit_comments(
        self,
        subreddit_names: list[str],
        callback: Callable | None = None,
        save_to_db: bool = True,
        skip_existing: bool = True
    ):
        """
        Stream new comments from specified subreddits.
        
        Args:
            subreddit_names: List of subreddit names (without 'r/')
            callback: Optional callback function for each comment
            save_to_db: Whether to save comments to database
            skip_existing: Skip already processed comments
        """
        if not self.listener:
            self.setup_listener(callback=callback, save_to_db=save_to_db)
        
        subreddit_str = '+'.join(subreddit_names)
        subreddit = self.reddit.subreddit(subreddit_str)
        
        logger.info(f"Starting Reddit comment stream for: r/{subreddit_str}")
        
        try:
            for comment in subreddit.stream.comments(skip_existing=skip_existing):
                self.listener.process_comment(comment)
        except Exception as e:
            logger.error(f"Error in comment stream: {e}")
    
    def stream_subreddit_all(
        self,
        subreddit_names: list[str],
        callback: Callable | None = None,
        save_to_db: bool = True,
        skip_existing: bool = True,
        stream_type: str = 'both'
    ):
        """
        Stream both submissions and comments from subreddits.
        
        Args:
            subreddit_names: List of subreddit names
            callback: Optional callback function
            save_to_db: Whether to save to database
            skip_existing: Skip already processed items
            stream_type: 'submissions', 'comments', or 'both'
        """
        if not self.listener:
            self.setup_listener(callback=callback, save_to_db=save_to_db)
        
        if stream_type in ['submissions', 'both']:
            logger.info("Starting submission stream...")
            self.stream_subreddit_submissions(
                subreddit_names, callback, save_to_db, skip_existing
            )
        
        if stream_type in ['comments', 'both']:
            logger.info("Starting comment stream...")
            self.stream_subreddit_comments(
                subreddit_names, callback, save_to_db, skip_existing
            )
    
    def get_hot_posts(
        self,
        subreddit_names: list[str],
        limit: int = 100,
        callback: Callable | None = None,
        save_to_db: bool = True
    ):
        """
        Get hot posts from subreddits (one-time fetch, not streaming).
        
        Useful for initial data collection or batch processing.
        """
        if not self.listener:
            self.setup_listener(callback=callback, save_to_db=save_to_db)
        
        subreddit_str = '+'.join(subreddit_names)
        subreddit = self.reddit.subreddit(subreddit_str)
        
        logger.info(f"Fetching {limit} hot posts from r/{subreddit_str}")
        
        for submission in subreddit.hot(limit=limit):
            self.listener.process_submission(submission)
    
    def get_new_posts(
        self,
        subreddit_names: list[str],
        limit: int = 100,
        callback: Callable | None = None,
        save_to_db: bool = True
    ):
        """Get newest posts from subreddits (one-time fetch)."""
        if not self.listener:
            self.setup_listener(callback=callback, save_to_db=save_to_db)
        
        subreddit_str = '+'.join(subreddit_names)
        subreddit = self.reddit.subreddit(subreddit_str)
        
        logger.info(f"Fetching {limit} new posts from r/{subreddit_str}")
        
        for submission in subreddit.new(limit=limit):
            self.listener.process_submission(submission)


def create_batch_collector(batch_size: int = 100):
    """
    Create a batch collector for efficient post processing.
    
    Useful for batching posts before sending to model for sentiment analysis.
    
    Args:
        batch_size: Number of posts to collect before processing
        
    Returns:
        Callable that collects posts and returns batch when size is reached
    """
    batch = []
    
    def collect_batch(post_data: dict):
        batch.append(post_data)
        if len(batch) >= batch_size:
            logger.info(f"Batch of {len(batch)} posts ready for processing")
            processed_batch = batch.copy()
            batch.clear()
            return processed_batch
        return None
    
    return collect_batch
