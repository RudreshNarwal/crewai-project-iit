"""Streamlit UI. Run: streamlit run app.py"""
import os

import streamlit as st

import assistant  # also loads .env
import rag

LABELS = {"pdf": "PDF Analyst", "web": "Web Researcher", "weather": "Weather Reporter"}

st.set_page_config(page_title="Agentic RAG with CrewAI", page_icon="🧭")
st.title("Agentic RAG with CrewAI")
st.caption("A Manager agent sends each question to a PDF, Web Search or Weather specialist.")

missing = [key for key in ("OPENROUTER_API_KEY", "OPENWEATHER_API_KEY") if not os.getenv(key)]
if missing:
    st.error(f"Missing {' and '.join(missing)}. Copy `.env.example` to `.env` and fill it in.")
    st.stop()

with st.sidebar:
    pdf = st.file_uploader("Upload a PDF", type="pdf")
    if not pdf:
        st.session_state.pop("collection", None)
        st.session_state.pop("pdf_id", None)
    elif st.session_state.get("pdf_id") != pdf.file_id:  # index once per upload, not per rerun
        try:
            with st.spinner("Indexing PDF..."):
                st.session_state.collection = rag.build_index(rag.read_pdf(pdf))
            st.session_state.pdf_id = pdf.file_id
        except ValueError as error:
            st.error(str(error))
    if "collection" in st.session_state:
        st.success(f"PDF indexed. Chunks: {st.session_state.collection.count()}")
    st.markdown("**Try**\n- What's the weather in Delhi?\n- Latest news on CrewAI\n- Summarise my PDF and tell me the weather in Mumbai")


def show(message: dict) -> None:
    with st.chat_message(message["role"]):
        if "routes" in message:
            routed = " + ".join(LABELS[route] for route in message["routes"])
            st.info(f"**Routed to:** {routed}\n\n{message['reason']}")
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                st.markdown("\n".join(f"- {source}" for source in message["sources"]))


messages = st.session_state.setdefault("messages", [])
for message in messages:
    show(message)

if question := st.chat_input("Ask about your PDF, the web, or the weather"):
    messages.append({"role": "user", "content": question})
    show(messages[-1])
    try:
        with st.spinner("Routing and answering..."):
            result = assistant.answer(question, st.session_state.get("collection"))
        messages.append({"role": "assistant", "content": result["answer"], **result})
    except Exception as error:  # bad key, no credit, network down: show it, keep the chat alive
        messages.append({"role": "assistant", "content": f"Something went wrong: {error}"})
    show(messages[-1])
