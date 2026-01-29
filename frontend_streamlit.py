import streamlit as st
import requests
import json
import os

# === CONFIG ===
st.set_page_config(page_title="🧠 Medical RAG Chatbot", layout="wide")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/query")

# === STYLES ===
st.markdown("""
    <style>
    [data-testid=stSidebar] {
        background-color: #0e1117;
        color: white;
    }
    .stSidebar header, .stSidebar h2, .stSidebar h3, .stSidebar p, .stSidebar span {
        color: white !important;
    }
    .stSidebar button {
        border-radius: 8px;
        background-color: #2e7bcf !important;
        color: white !important;
        border: none;
    }
    .stSidebar button:hover {
        background-color: #5294e2 !important;
    }
    .stChatMessage {
        border-radius: 15px;
        margin: 10px 0;
    }
    .user-msg {
        background-color: #1e1e1e;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
    }
    .bot-msg {
        background-color: #2b313e;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
    }
    .icon {
        font-size: 20px;
        margin-right: 8px;
        color: #5294e2;
    }
    </style>
""", unsafe_allow_html=True)

# === SIMPLE USER LOGIN SYSTEM ===
def user_auth():
    st.sidebar.markdown("<h2><i class='icon'>👤</i> User Login / Sign Up</h2>", unsafe_allow_html=True)
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if "users" not in st.session_state:
        st.session_state["users"] = {}

    if st.sidebar.button("Sign Up"):
        if username and password:
            st.session_state["users"][username] = {"password": password, "records": []}
            st.sidebar.success("✅ Account created! Please log in.")
        else:
            st.sidebar.error("Please enter username and password.")

    if st.sidebar.button("Login"):
        if username in st.session_state["users"] and st.session_state["users"][username]["password"] == password:
            st.session_state["user"] = username
            st.sidebar.success(f"Welcome, {username} 👋")
        else:
            st.sidebar.error("Invalid credentials.")

    return st.session_state.get("user", None)


# === HEALTH RECORDS SYSTEM ===
def show_health_records(user):
    st.sidebar.markdown("<h3><i class='icon'>🩺</i> Health Records</h3>", unsafe_allow_html=True)
    if user in st.session_state["users"]:
        records = st.session_state["users"][user]["records"]

        if records:
            for record in records:
                st.sidebar.markdown(f"<p>📅 <b>{record['date']}</b>: {record['note']}</p>", unsafe_allow_html=True)
        else:
            st.sidebar.info("No health records yet.")

        st.sidebar.markdown("<hr>", unsafe_allow_html=True)
        new_note = st.sidebar.text_input("Add new record note")
        if st.sidebar.button("Add Record"):
            if new_note:
                st.session_state["users"][user]["records"].append(
                    {"date": "Today", "note": new_note}
                )
                st.sidebar.success("✅ Record added!")


# === CHAT INTERFACE ===
def chatbot_ui():
    st.markdown("<h1 style='text-align:center; color:#2e7bcf;'>💬 Medical RAG Chatbot Assistant</h1>", unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_query = st.chat_input("Ask a medical question...")

    if user_query:
        st.session_state["chat_history"].append({"role": "user", "content": user_query})

        try:
            payload = {"query": user_query, "top_k": 3}
            response = requests.post(API_URL, json=payload, timeout=60)

            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer found.")
                contexts = data.get("contexts", [])

                st.session_state["chat_history"].append({"role": "bot", "content": answer, "contexts": contexts})
            else:
                st.session_state["chat_history"].append(
                    {"role": "bot", "content": f"⚠️ API Error: {response.status_code} - {response.text}"}
                )

        except Exception as e:
            st.session_state["chat_history"].append(
                {"role": "bot", "content": f"❌ Could not connect to backend: {e}"}
            )

    for chat in st.session_state["chat_history"]:
        if chat["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"<div class='user-msg'>🙋‍♂️ <b>You:</b> {chat['content']}</div>", unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.markdown(f"<div class='bot-msg'>🤖 <b>Assistant:</b> {chat['content']}</div>", unsafe_allow_html=True)
                if "contexts" in chat and chat["contexts"]:
                    with st.expander("📘 Relevant Medical References"):
                        for ctx in chat["contexts"]:
                            st.markdown(f"- {ctx}")


# === MAIN APP FLOW ===
user = user_auth()
if user:
    show_health_records(user)
    chatbot_ui()
else:
    st.warning("🔒 Please log in or sign up to access the chatbot.")
