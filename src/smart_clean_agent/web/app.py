import sys
from pathlib import Path

import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smart_clean_agent.web.bootstrap import build_app_context, validate_online_dependencies
from smart_clean_agent.rag.vector_store import get_vector_store_directory
from smart_clean_agent.ui.chat_components import handle_chat_interaction, render_chat_history
from smart_clean_agent.ui.sidebar_components import render_session_sidebar, render_user_profile_sidebar


st.title("智扫通机器人智能客服")
st.divider()

try:
    validate_online_dependencies()
except RuntimeError as exc:
    st.error(str(exc))
    st.info("请先补齐环境变量与本地向量库，再执行 `streamlit run src/smart_clean_agent/web/app.py`。如未建库，请先运行 `python src/smart_clean_agent/rag/ingest.py`。")
    st.caption(f"当前向量库存储目录: {get_vector_store_directory()}")
    st.stop()

try:
    app_context = build_app_context(st.session_state)
except Exception as exc:
    st.sidebar.error(f"应用初始化失败: {str(exc)}")
    st.stop()

with st.sidebar:
    render_user_profile_sidebar(
        user_profiles=app_context.user_profiles,
        selected_profile=app_context.selected_profile,
        session_state=st.session_state,
    )
    render_session_sidebar(st.session_state)

render_chat_history(st.session_state)
handle_chat_interaction(st.session_state)

