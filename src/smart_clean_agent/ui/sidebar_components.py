from typing import Any, MutableMapping

import streamlit as st

from smart_clean_agent.web.bootstrap import sync_session_state
from smart_clean_agent.services.session_service import create_session, delete_session, list_sessions, load_session
from smart_clean_agent.services.user_profile_service import upsert_user_profile


SessionState = MutableMapping[str, Any]


def render_user_profile_sidebar(
    user_profiles: list[dict[str, str]],
    selected_profile: dict[str, str],
    session_state: SessionState,
) -> None:
    user_options = [profile["user_id"] for profile in user_profiles]

    st.subheader("用户与会话")
    selected_user_id = st.selectbox("选择用户ID", options=user_options, key="selected_user_id")
    if selected_user_id != selected_profile["user_id"]:
        st.rerun()

    with st.expander("用户信息", expanded=False):
        with st.form("edit_user_profile_form"):
            st.text_input("用户ID", value=selected_profile["user_id"], disabled=True)
            edit_city = st.text_input("城市", value=selected_profile.get("city", ""))
            edit_name = st.text_input("用户名称", value=selected_profile.get("name", ""))
            edit_house_type = st.text_input("户型信息", value=selected_profile.get("house_type", ""))
            edit_floor_type = st.text_input("地面类型", value=selected_profile.get("floor_type", ""))
            if st.form_submit_button("保存当前用户资料", use_container_width=True):
                try:
                    updated_profile = upsert_user_profile(
                        {
                            "user_id": selected_profile["user_id"],
                            "city": edit_city,
                            "name": edit_name,
                            "house_type": edit_house_type,
                            "floor_type": edit_floor_type,
                        },
                        original_user_id=selected_profile["user_id"],
                    )
                    session_state["selected_city"] = updated_profile["city"]
                    st.success("用户资料已保存")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

    with st.expander("新增用户"):
        with st.form("add_user_profile_form"):
            new_user_id = st.text_input("新用户ID")
            new_city = st.text_input("新用户城市")
            new_name = st.text_input("新用户名称")
            new_house_type = st.text_input("新用户户型信息")
            new_floor_type = st.text_input("新用户地面类型")
            if st.form_submit_button("新增用户", use_container_width=True):
                try:
                    upsert_user_profile(
                        {
                            "user_id": new_user_id,
                            "city": new_city,
                            "name": new_name,
                            "house_type": new_house_type,
                            "floor_type": new_floor_type,
                        }
                    )
                    session_state["selected_user_id"] = new_user_id.strip()
                    st.success("新用户已创建")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))


def render_session_sidebar(session_state: SessionState) -> None:
    current_user_id = session_state["selected_user_id"]

    st.subheader("会话管理")
    if st.button("新建会话", use_container_width=True):
        session_data = create_session(current_user_id)
        sync_session_state(session_state, current_user_id, session_data)
        st.rerun()

    session_list = list_sessions(current_user_id)
    for session in session_list:
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(
                session["title"],
                key=f"load_{current_user_id}_{session['session_id']}",
                use_container_width=True,
                type="primary" if session["session_id"] == session_state.get("current_session_id") else "secondary",
            ):
                loaded_session = load_session(current_user_id, session["session_id"])
                if loaded_session is not None:
                    sync_session_state(session_state, current_user_id, loaded_session)
                    st.rerun()

        with col2:
            if st.button("删", key=f"delete_{current_user_id}_{session['session_id']}", use_container_width=True):
                delete_session(current_user_id, session["session_id"])
                remaining_sessions = list_sessions(current_user_id)
                if session_state.get("current_session_id") == session["session_id"]:
                    next_session = remaining_sessions[0] if remaining_sessions else create_session(current_user_id)
                    sync_session_state(session_state, current_user_id, next_session)
                st.rerun()

    st.divider()
    st.caption("当前会话摘要")
    st.caption(session_state.get("current_session_summary") or "暂无会话摘要")

