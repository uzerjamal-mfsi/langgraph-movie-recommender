from langchain_core.messages import AIMessage, HumanMessage

INTENT_ERROR = "I couldn't understand your request. Please try again later."
GENERAL_ERROR = "Something went wrong please try again."
FEEDBACK_EXTRACTION_ERROR = "I couldn't process your feedback. Please try again later."
PREFERENCE_EXTRACTION_ERROR = "I couldn't process your preferences. Please try again later."
SAVE_FEEDBACK_ERROR = "I couldn't save your feedback. Please try again later."
SAVE_PREFERENCES_ERROR = "I couldn't save your preferences. Please try again later."
TITLE_LOOKUP_ERROR = "I couldn't look up that title right now. Please try again later."


def get_latest_user_message(state):
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content

    return ""


def failed(message, **extra):
    return {"error": message, **extra}


def error_result(message):
    return {"messages": [AIMessage(content=message)], "error": message}
