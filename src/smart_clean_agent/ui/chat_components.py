import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from typing import Any, MutableMapping

import streamlit as st

from smart_clean_agent.web.bootstrap import sync_session_state
from smart_clean_agent.services.chat_service import ChatServiceError, run_chat


SessionState = MutableMapping[str, Any]
PROCESS_PANEL_CSS = """
<style>
.process-note-wrapper {
  color: #9ca3af;
  font-size: 0.9rem;
  line-height: 1.85;
}
.process-note-loading {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  margin-bottom: 0.45rem;
}
.process-note-spinner {
  width: 0.82rem;
  height: 0.82rem;
  border: 2px solid rgba(156, 163, 175, 0.3);
  border-top-color: #9ca3af;
  border-radius: 50%;
  animation: process-note-spin 0.9s linear infinite;
  flex-shrink: 0;
}
.process-note-item {
  margin: 0.18rem 0;
}
@keyframes process-note-spin {
  to { transform: rotate(360deg); }
}
</style>
"""


def render_chat_history(session_state: SessionState) -> None:
    for message in session_state["message"]:
        st.chat_message(message["role"]).write(message["content"])


def handle_chat_interaction(session_state: SessionState) -> None:
    prompt = st.chat_input()
    if not prompt:
        return

    st.chat_message("user").write(prompt)
    current_process_events: list[dict[str, str]] = []
    event_queue: Queue[dict[str, str]] = Queue()

    try:
        with st.chat_message("assistant"):
            with st.expander("查看处理过程", expanded=False):
                process_placeholder = st.empty()

            def render_process_notes(is_processing: bool = True) -> None:
                notes = build_process_notes(current_process_events)
                process_placeholder.markdown(
                    build_process_panel_html(notes, is_processing=is_processing),
                    unsafe_allow_html=True,
                )

            def on_status_event(event: dict[str, str]) -> None:
                event_queue.put(event)

            render_process_notes()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    run_chat,
                    user_id=session_state["selected_user_id"],
                    message=prompt,
                    session_id=session_state.get("current_session_id"),
                    status_event_callback=on_status_event,
                )

                while not future.done():
                    _drain_process_event_queue(event_queue, current_process_events)
                    render_process_notes()
                    time.sleep(0.05)

                _drain_process_event_queue(event_queue, current_process_events)
                render_process_notes(is_processing=False)
                result = future.result()
            st.write_stream(_yield_response_chunks(result["answer"]))
    except ChatServiceError as exc:
        st.error(str(exc))
        return

    session_state["latest_status_events"] = result["status_events"]
    sync_session_state(session_state, session_state["selected_user_id"], result["session_data"])
    st.rerun()


def build_process_notes(events: list[dict[str, str]]) -> list[str]:
    notes: list[str] = []
    for event in events:
        note = _build_process_note(event)
        if note and note not in notes:
            notes.append(note)
    return notes


def build_process_panel_html(notes: list[str], is_processing: bool = True) -> str:
    loading_line = (
        "<div class='process-note-loading'>"
        "<span class='process-note-spinner'></span>"
        "<span>我正在整理信息，请稍等片刻...</span>"
        "</div>"
        if is_processing
        else "<div class='process-note-item'>本轮处理过程已完成。</div>"
    )

    note_lines = notes or ["正在准备处理你的问题..."]
    notes_html = "".join(
        f"<div class='process-note-item'>{index}. {note}</div>"
        for index, note in enumerate(note_lines, start=1)
    )
    return f"{PROCESS_PANEL_CSS}<div class='process-note-wrapper'>{loading_line}{notes_html}</div>"


def _yield_response_chunks(text: str, chunk_size: int = 4, delay: float = 0.02):
    buffer = ""
    for char in text:
        buffer += char
        if char == "\n" or len(buffer) >= chunk_size:
            yield buffer
            time.sleep(delay)
            buffer = ""

    if buffer:
        yield buffer


def _drain_process_event_queue(
    event_queue: Queue[dict[str, str]],
    current_process_events: list[dict[str, str]],
) -> None:
    while True:
        try:
            event = event_queue.get_nowait()
        except Empty:
            return
        current_process_events.append(event)


def _build_process_note(event: dict[str, str]) -> str:
    event_type = (event.get("event_type") or "").strip()
    detail = (event.get("detail") or "").strip()

    if event_type == "stage.memory":
        return "我先结合你最近的会话、长期偏好和趋势记录看一下。"
    if event_type == "stage.model":
        return "我在判断这次需要补哪些信息，才能给你更准确的答复。"
    if event_type == "stage.rag":
        return "我在检索和扫地机器人使用相关的知识资料。"
    if event_type == "stage.report":
        return "我在切换到报告生成模式，准备整理你的使用报告。"
    if event_type == "stage.trend":
        return "我在汇总最近几个月的使用趋势变化。"
    if event_type == "stage.final":
        return "我已经整理好结论，马上给你正式答复。"
    if event_type == "error.tool":
        return "工具调用出现了一点问题，我正在尽量整理仍然可用的信息。"
    if event_type == "error.agent":
        return "当前处理过程中出现异常，本次回复可能无法完整生成。"
    if event_type != "stage.tool":
        return ""

    if "get_weather" in detail:
        return "我在查询你所在城市的天气情况。"
    if "get_user_location" in detail:
        return "我在确认你当前所在的城市。"
    if "get_user_id" in detail:
        return "我在确认当前用户身份。"
    if "get_current_month" in detail:
        return "我在确认当前需要查询的报告月份。"
    if "fetch_external_data" in detail:
        return "我在读取你当月的使用记录。"
    if "fetch_external_history" in detail:
        return "我在补充最近几个月的趋势记录。"
    return "我在补充外部信息，确保回答更完整。"

