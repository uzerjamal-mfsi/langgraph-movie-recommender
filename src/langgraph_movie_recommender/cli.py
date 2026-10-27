import logging
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from langgraph_movie_recommender.config import get_settings
from langgraph_movie_recommender.db.session import init_db
from langgraph_movie_recommender.graph.workflow import build_workflow
from langgraph_movie_recommender.log_config import configure_logging

GREETING_MESSAGE = """Movie & TV Recommender
Type your message or 'exit' to quit.
"""
EXIT_MESSAGE = "Exiting.. Goodbye!"
EXIT_COMMANDS = {"exit", "quit"}
MAX_MESSAGE_LENGTH = 2000
PROMPT_LABEL = "You: "
STREAMING_NODE = "answer_general"
STREAM_MODES = ["messages", "values"]

logger = logging.getLogger(__name__)


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


def response_text(message):
    """Extract plain text from a message whose content may be a list of blocks."""
    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)

    return ""


def stream_turn(workflow, payload, config):
    """Stream answer_general tokens. Returns True when anything was printed."""
    streamed = False

    for mode, chunk in workflow.stream(payload, config=config, stream_mode=STREAM_MODES):
        if mode != "messages":
            continue

        message, metadata = chunk
        if (metadata or {}).get("langgraph_node") != STREAMING_NODE:
            continue

        text = response_text(message)
        if not text:
            continue

        if not streamed:
            print("Assistant: ", end="", flush=True)
            streamed = True

        print(text, end="", flush=True)

    if streamed:
        print()

    return streamed


def run_chat(thread_id=None):
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db()
    settings.checkpoint_db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if settings.langsmith_tracing:
        logger.info("LangSmith tracing enabled")
    session_thread_id = resolve_thread_id(thread_id)
    config = {
        "configurable": {"thread_id": session_thread_id},
        "run_name": "movie_recommender_chat",
    }
    waiting_for_clarification = False

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

            if waiting_for_clarification:
                payload = Command(resume=message)
            else:
                payload = {"messages": [HumanMessage(content=message)]}

            try:
                streamed = stream_turn(workflow, payload, config)
            except KeyboardInterrupt:
                print()
                break
            except Exception:
                logger.exception("run_chat: failed to process message")
                print("Assistant: Something went wrong. Please try again.")
                continue

            snapshot = workflow.get_state(config)
            interruptions = snapshot.interrupts

            if interruptions:
                waiting_for_clarification = True
                for interruption in interruptions:
                    print(f"Assistant: {interruption.value}")
                continue

            waiting_for_clarification = False

            if not streamed:
                messages = snapshot.values.get("messages") or []
                if messages:
                    print(f"Assistant: {response_text(messages[-1])}")
                else:
                    print("Assistant: I did not produce a response. Please try again")

    print(EXIT_MESSAGE)
    return 0
