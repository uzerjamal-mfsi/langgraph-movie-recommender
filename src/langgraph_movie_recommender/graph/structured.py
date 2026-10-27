import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

REPAIR_INSTRUCTION = """\
Your previous reply could not be parsed into the required schema.
Reply with a single JSON object only. Do not include prose, markdown fences, or comments.
The object must validate against this JSON schema:
{schema}
"""


def message_text(message: Any) -> str:
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
        return "\n".join(parts)

    return ""


def extract_json_payload(text: str) -> Any:
    """Parse JSON from a model reply, tolerating fences and surrounding prose."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character not in "{[":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        return payload

    raise ValueError("No JSON object found in model response.")


def _coerce_result[T: BaseModel](result: Any, schema: type[T]) -> T:
    if isinstance(result, schema):
        return result
    if isinstance(result, BaseModel):
        return schema.model_validate(result.model_dump())
    return schema.model_validate(result)


def invoke_structured[T: BaseModel](
    model: BaseChatModel,
    schema: type[T],
    messages: list[BaseMessage],
    *,
    operation: str,
) -> T | None:
    """Invoke structured output, falling back to plain JSON on parser failures.

    Providers occasionally fail function calling or emit a payload that does not
    validate. The fallback asks for the same schema as JSON and parses it locally.
    Returns None when both paths fail; callers keep their existing user-facing error.
    """
    try:
        return _coerce_result(model.with_structured_output(schema).invoke(messages), schema)
    except Exception:
        logger.warning(
            "%s: structured output failed; retrying with a JSON fallback",
            operation,
            exc_info=True,
        )

    repair_messages = [
        *messages,
        HumanMessage(
            content=REPAIR_INSTRUCTION.format(
                schema=json.dumps(schema.model_json_schema(), indent=2),
            )
        ),
    ]

    try:
        response = model.invoke(repair_messages)
        payload = extract_json_payload(message_text(response))
        return schema.model_validate(payload)
    except Exception:
        logger.exception("%s: JSON fallback failed", operation)
        return None
