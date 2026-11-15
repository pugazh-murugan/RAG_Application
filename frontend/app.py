import json
import os
import uuid

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Uzhavan RAG Assistant", page_icon="🌾", layout="wide")


def new_chat() -> str:
    chat_id = uuid.uuid4().hex[:8]
    st.session_state.chats[chat_id] = {"title": "New chat", "messages": []}
    st.session_state.current = chat_id
    return chat_id


if "chats" not in st.session_state:
    st.session_state.chats = {}
    new_chat()
st.session_state.setdefault("uploader_key", 0)


def api_error(resp: requests.Response) -> str:
    try:
        return resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text


def fetch_documents() -> list[dict]:
    try:
        resp = requests.get(f"{BACKEND_URL}/documents", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"Cannot reach backend: {exc}")
        return []


def render_sources(sources: list[str]) -> None:
    if not sources:
        return
    lines = [f"- [{s}]({s})" if s.startswith("http") else f"- {s}" for s in sources]
    st.markdown("**Sources**\n" + "\n".join(lines))


def stream_chat(question: str, holder: dict):
    try:
        with requests.post(
            f"{BACKEND_URL}/chat", json={"question": question, "stream": True}, stream=True, timeout=120
        ) as resp:
            if resp.status_code != 200:
                yield f"⚠️ {api_error(resp)}"
                return
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                event = json.loads(line)
                if event["type"] == "token":
                    yield event["text"]
                elif event["type"] == "sources":
                    holder["sources"] = event["sources"]
                elif event["type"] == "error":
                    yield f"⚠️ {event['message']}"
    except requests.RequestException as exc:
        yield f"⚠️ Cannot reach backend: {exc}"


chat_col, side_col = st.columns([1.4, 1], gap="large")

with chat_col:
    st.title("🌾 Uzhavan Assistant")
    top_a, top_b = st.columns([1, 1])
    with top_a:
        if st.button("➕ New chat", width="stretch"):
            new_chat()
            st.rerun()
    with top_b:
        with st.popover("🕘 Chat history", width="stretch"):
            for cid, chat in reversed(list(st.session_state.chats.items())):
                label = ("● " if cid == st.session_state.current else "") + chat["title"]
                if st.button(label, key=f"hist_{cid}", width="stretch"):
                    st.session_state.current = cid
                    st.rerun()

    current = st.session_state.chats[st.session_state.current]
    messages_box = st.container(height=520, border=True)
    with messages_box:
        if not current["messages"]:
            st.caption("Ask about schemes, subsidies, crop advisories or prices — in Tamil or English.")
        for msg in current["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    render_sources(msg.get("sources", []))

    prompt = st.chat_input("Ask a question / கேள்வியை உள்ளிடவும்")
    if prompt:
        if not current["messages"]:
            current["title"] = prompt[:40]
        current["messages"].append({"role": "user", "content": prompt})
        with messages_box:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                holder: dict = {"sources": []}
                answer = st.write_stream(stream_chat(prompt, holder))
                render_sources(holder["sources"])
        current["messages"].append({"role": "assistant", "content": answer, "sources": holder["sources"]})

with side_col:
    st.subheader("📁 Knowledge base")
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))
    files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "csv", "xlsx"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if st.button("Upload & Index", type="primary", disabled=not files, width="stretch"):
        done = 0
        for f in files:
            with st.spinner(f"Indexing {f.name}…"):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/upload/file",
                        files={"file": (f.name, f.getvalue())},
                        timeout=600,
                    )
                    if resp.ok:
                        done += 1
                    else:
                        st.error(f"{f.name}: {api_error(resp)}")
                except requests.RequestException as exc:
                    st.error(f"{f.name}: {exc}")
        if done:
            st.session_state.flash = f"Indexed {done} file(s)"
            st.session_state.uploader_key += 1
            st.rerun()

    st.markdown("##### Indexed files")
    docs = fetch_documents()
    if docs:
        df = pd.DataFrame(docs)[["source_name", "source_type", "chunks", "id"]]
        st.dataframe(df, hide_index=True, width="stretch", height=220)
        options = {f"{d['source_name']} ({d['source_type']})": d["id"] for d in docs}
        choice = st.selectbox("Remove a source", ["—"] + list(options), label_visibility="collapsed")
        if st.button("🗑️ Delete selected", disabled=choice == "—", width="stretch"):
            resp = requests.delete(f"{BACKEND_URL}/documents/{options[choice]}", timeout=60)
            if resp.ok:
                st.rerun()
            else:
                st.error(api_error(resp))
    else:
        st.caption("No files indexed yet.")

    st.divider()
    st.markdown("##### Uzhavan website")
    if st.button("🌐 Scrape Uzhavan website", width="stretch"):
        with st.spinner("Scraping pages and indexing… this can take a few minutes"):
            try:
                resp = requests.post(f"{BACKEND_URL}/scrape/uzhavan", timeout=1800)
                if resp.ok:
                    data = resp.json()
                    st.session_state.flash = (
                        f"Scraped {data['pages_scraped']} pages → {data['chunks']} chunks indexed"
                    )
                    st.rerun()
                else:
                    st.error(api_error(resp))
            except requests.RequestException as exc:
                st.error(f"Scrape failed: {exc}")

    excel_bytes = None
    try:
        r = requests.get(f"{BACKEND_URL}/download/excel", timeout=60)
        if r.ok:
            excel_bytes = r.content
    except requests.RequestException:
        pass
    st.download_button(
        "⬇️ Download Excel",
        data=excel_bytes or b"",
        file_name="uzhavan_pages.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=excel_bytes is None,
        width="stretch",
    )
