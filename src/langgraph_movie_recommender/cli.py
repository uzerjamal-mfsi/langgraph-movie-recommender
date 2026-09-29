import logging
import uuid

from langgraph_movie_recommender.config import get_settings

GREETING_MESSAGE = """Movie & TV Recommender
Type your message or 'exit' to quit.
"""
EXIT_MESSAGE = "Exiting.. Goodbye!"
EXIT_COMMANDS = {"exit", "quit"}
MAX_MESSAGE_LENGTH = 2000
PROMPT_LABEL = "You: "


def create_thread_id():
    return uuid.uuid4().hex


def resolve_thread_id(thread_id):
    if thread_id is None:
        return create_thread_id()
    
    cleaned = thread_id.strip()
    return cleaned


def read_message():
    try:
        raw_message = input(PROMPT_LABEL)
    except (KeyboardInterrupt):
        return None

    message = raw_message.strip()
    if not message:
        print("Please type a message, or 'exit' to quit.")
        return ""
    if message.lower() in EXIT_COMMANDS:
        return None
    if len(message) > MAX_MESSAGE_LENGTH:
        print(f"Message is too long. Please keep it under {MAX_MESSAGE_LENGTH} characters.")
        return ""
    return message


def run_chat(thread_id=None):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    session_thread_id = resolve_thread_id(thread_id)

    print(GREETING_MESSAGE)
    print(f"Session thread: {session_thread_id}")

    while True:
        message = read_message()
        if message is None:
            break
        if message:
            print(f"Assistant: You said: {message}")

    print(EXIT_MESSAGE)
    return 0
