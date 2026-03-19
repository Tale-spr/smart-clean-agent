from typing import Iterable


MAX_TEXT_LENGTH = 120



def normalize_message_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= max_length:
        return normalized

    return normalized[: max_length - 3] + "..."



def _iter_messages(messages: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    valid_messages: list[dict[str, str]] = []
    for message in messages:
        role = (message.get("role") or "").strip()
        content = (message.get("content") or "").strip()
        if not role or not content:
            continue
        valid_messages.append({"role": role, "content": content})
    return valid_messages



def build_recent_history(messages: list[dict[str, str]], max_messages: int = 6) -> str:
    valid_messages = _iter_messages(messages)
    if not valid_messages:
        return ""

    recent_messages = valid_messages[-max_messages:]
    lines: list[str] = []
    for message in recent_messages:
        speaker = "用户" if message["role"] == "user" else "助手"
        lines.append(f"{speaker}: {normalize_message_text(message['content'])}")

    return "\n".join(lines)



def summarize_messages(messages: list[dict[str, str]], max_user_points: int = 3, max_assistant_points: int = 2) -> str:
    valid_messages = _iter_messages(messages)
    if not valid_messages:
        return ""

    recent_messages = valid_messages[-8:]
    user_messages = [normalize_message_text(message["content"]) for message in recent_messages if message["role"] == "user"]
    assistant_messages = [normalize_message_text(message["content"]) for message in recent_messages if message["role"] == "assistant"]

    parts: list[str] = []
    if user_messages:
        parts.append("最近用户关注: " + "；".join(user_messages[-max_user_points:]))
    if assistant_messages:
        parts.append("最近已提供的信息: " + "；".join(assistant_messages[-max_assistant_points:]))

    return "\n".join(parts)

