#!/usr/bin/env python3
"""
Reddit Streaming Script

Example script to stream Reddit data for sentiment analysis.
Completely free - no credit card required!
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_ingestion.reddit_streamer import RedditStreamManager
from config import settings


def main():
    parser = argparse.ArgumentParser(description='Stream Reddit data for sentiment analysis')
    parser.add_argument(
        '--subreddits',
        nargs='+',
        default=['technology', 'news', 'worldnews'],
        help='Subreddits to monitor (without r/)'
    )
    parser.add_argument(
        '--type',
        choices=['submissions', 'comments', 'both'],
        default='submissions',
        help='Type of content to stream'
    )
    parser.add_argument(
        '--save-db',
        action='store_true',
        default=True,
        help='Save posts to database'
    )
    parser.add_argument(
        '--batch-fetch',
        action='store_true',
        help='Fetch hot posts instead of streaming (for initial data collection)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Number of posts to fetch in batch mode'
    )
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Reddit data collection...")
    print(f"📍 Subreddits: r/{', r/'.join(args.subreddits)}")
    print(f"📝 Type: {args.type}")
    print(f"💾 Save to DB: {args.save_db}")
    print()
    
    manager = RedditStreamManager()
    
    if args.batch_fetch:
        print(f"📦 Fetching {args.limit} hot posts...")
        manager.get_hot_posts(
            subreddit_names=args.subreddits,
            limit=args.limit,
            save_to_db=args.save_db
        )
        print("✅ Batch fetch completed!")
    else:
        print("🔄 Starting real-time stream (Press Ctrl+C to stop)...")
        try:
            if args.type == 'submissions':
                manager.stream_subreddit_submissions(
                    subreddit_names=args.subreddits,
                    save_to_db=args.save_db
                )
            elif args.type == 'comments':
                manager.stream_subreddit_comments(
                    subreddit_names=args.subreddits,
                    save_to_db=args.save_db
                )
            else:
                manager.stream_subreddit_all(
                    subreddit_names=args.subreddits,
                    save_to_db=args.save_db,
                    stream_type='both'
                )
        except KeyboardInterrupt:
            print("\n⏹️  Stream stopped by user")
            print("✅ Exiting gracefully...")


if __name__ == "__main__":
    main()
