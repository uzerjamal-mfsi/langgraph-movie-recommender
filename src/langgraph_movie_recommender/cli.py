import logging
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph_movie_recommender.config import get_settings
from langgraph_movie_recommender.db.session import init_db
from langgraph_movie_recommender.graph.workflow import build_workflow

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
    except KeyboardInterrupt:
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
    init_db()
    settings = get_settings()
    settings.checkpoint_db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    logging.basicConfig(level=settings.log_level)
    session_thread_id = resolve_thread_id(thread_id)
    config = {"configurable": {"thread_id": session_thread_id}}

    with SqliteSaver.from_conn_string(str(settings.checkpoint_db_path)) as checkpointer:
        workflow = build_workflow(checkpointer=checkpointer)

        print(GREETING_MESSAGE)
        print(f"Session thread: {session_thread_id}")

        while True:
            message = read_message()
            if message is None:
                break
            if not message:
                continue

            try:
                result = workflow.invoke(
                    {"messages": [HumanMessage(content=message)]},
                    config=config,
                )
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Assistant: ERROR: {e} \n Please try again.")
                continue

            messages = result.get("messages") or []
            if messages:
                print(f"Assistant: {messages[-1].content}")
            else:
                print("Assistant: I did not produce a response. Please try again")

    print(EXIT_MESSAGE)
    return 0
