def route_after_title_resolution(state):
    if state.get("error"):
        return "error"

    if state.get("needs_clarification"):
        return "clarify"

    return "save"


def route_after_clarification(state):
    if state.get("error"):
        return "error"

    if state.get("needs_clarification"):
        return "clarify"

    return "save"


def route_intent(state):
    if state.get("error"):
        return "error"

    return state.get("intent", "general")
