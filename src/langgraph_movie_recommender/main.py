import argparse
import logging

from langgraph_movie_recommender.cli import run_chat
from langgraph_movie_recommender.db.session import init_db
from langgraph_movie_recommender.memory import rebuild_preference_index


def main():
    parser = argparse.ArgumentParser(
        prog="langgraph-movie-recommender",
        description="Movie & TV Recommender Chatbot",
    )
    parser.add_argument("--thread-id", help="Resume an existing conversation thread")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the Qdrant preference index and exit",
    )
    arguments = parser.parse_args()

    if arguments.rebuild_index:
        logging.basicConfig(level=logging.INFO)
        init_db()
        count = rebuild_preference_index()
        print(f"Rebuilt preference index with {count} preferences.")
        return 0

    run_chat(thread_id=arguments.thread_id)
    return 0
