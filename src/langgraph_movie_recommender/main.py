import argparse

from langgraph_movie_recommender.cli import run_chat


def main():
    parser = argparse.ArgumentParser(
        prog="langgraph-movie-recommender",
        description="Movie & TV Recommender Chatbot",
    )
    parser.add_argument("--thread-id", help="Resume an existing conversation thread")
    arguments = parser.parse_args()

    run_chat(thread_id=arguments.thread_id)
