"""cdo-ai-platform: a Streamlit front end tying together CSV import, a data
workspace, and a Claude-powered copilot chat.

Run locally with: streamlit run app.py
"""

# --- Imports -----------------------------------------------------------------
import json
import os
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st

# --- Constants ---------------------------------------------------------------
MODEL = "claude-sonnet-4-5"
DATA_DIR = Path(__file__).parent / "data"  # where the Importer saves mapped output

CANONICAL_FIELDS = ("Customer", "Job", "Invoice", "Payment", "Task", "Vendor", "VendorID")

MAPPER_SYSTEM_PROMPT = f"""You are a data mapper. Given CSV/XLSX headers and sample values,
return a JSON array mapping each source column to a canonical field.

Canonical fields: {", ".join(CANONICAL_FIELDS)}, Unknown

Return ONLY a JSON array. No explanation. Each item must have:
- "source_column": the original header name
- "canonical_field": the best matching canonical field
- "confidence": a float between 0.0 and 1.0"""


# --- Shared helpers ------------------------------------------------------------

def get_client() -> anthropic.Anthropic | None:
    """Build an Anthropic client from an API key, wherever it comes from.

    Streamlit reruns this whole script top-to-bottom every time the user
    interacts with a widget, so this needs to be cheap to call repeatedly.
    Checks the ANTHROPIC_API_KEY environment variable first (e.g. set via a
    .env-loaded shell); falls back to the sidebar text input in
    render_sidebar() so the app still works the very first time, before
    anyone has set up a .env file. Returns None (rather than raising) when
    no key is available yet, so callers can show a friendly message instead
    of crashing the whole page.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY") or st.session_state.get("api_key")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def clean_json_response(text: str) -> str:
    """Strip optional ```json ... ``` fences before parsing a Claude reply as JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    return text


def load_dataframe(uploaded_file) -> pd.DataFrame:
    """Read a Streamlit-uploaded CSV or XLSX file into a DataFrame."""
    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


def list_mapped_files() -> list[Path]:
    """Every mapping the Importer has saved so far, newest first."""
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.mapped.json"), key=lambda p: p.stat().st_mtime, reverse=True)


# --- Sidebar: API key + page navigation ---------------------------------------

def render_sidebar() -> str:
    """Draw the sidebar and return which page ("Importer"/"Workspace"/"Copilot")
    the user has selected via st.sidebar.radio.
    """
    st.sidebar.title("CDO AI Platform")

    # Only prompt for a key if one isn't already sitting in the environment —
    # keeps local dev painless if ANTHROPIC_API_KEY is already exported.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.session_state.setdefault("api_key", "")
        st.session_state["api_key"] = st.sidebar.text_input(
            "Anthropic API key",
            value=st.session_state["api_key"],
            type="password",
            help="Only needed for Importer's mapping step and the Copilot chat.",
        )

    return st.sidebar.radio("Go to", ("Importer", "Workspace", "Copilot"))


# --- Page: Importer -------------------------------------------------------------

def render_importer():
    st.header("Importer")
    st.write("Upload a CSV or XLSX file and map its columns to canonical fields using Claude.")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    if uploaded_file is None:
        return

    df = load_dataframe(uploaded_file)
    st.subheader("Preview")
    st.dataframe(df.head())

    if st.button("Map columns with Claude"):
        client = get_client()
        if client is None:
            st.error("Set an Anthropic API key in the sidebar first.")
            return

        # First 3 non-null sample values per column give Claude enough
        # context to guess the right canonical field without sending the
        # whole file.
        samples = {col: df[col].dropna().astype(str).tolist()[:3] for col in df.columns}
        user_message = "Headers and sample values:\n" + "\n".join(
            f"- '{col}': {samples.get(col, [])}" for col in df.columns
        )

        with st.spinner("Asking Claude to map columns..."):
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=MAPPER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            mappings = json.loads(clean_json_response(response.content[0].text))

        # Stashed in session_state (rather than a local variable) so the
        # mapping table below survives the rerun that happens after this
        # button click and stays visible until a *different* file is picked.
        st.session_state["last_mappings"] = mappings
        st.session_state["last_filename"] = uploaded_file.name

    mappings = st.session_state.get("last_mappings")
    if mappings and st.session_state.get("last_filename") == uploaded_file.name:
        st.subheader("Mapping results")
        st.dataframe(pd.DataFrame(mappings))

        if st.button("Save mapping to Workspace"):
            DATA_DIR.mkdir(exist_ok=True)
            stem = Path(uploaded_file.name).stem
            out_path = DATA_DIR / f"{stem}.mapped.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(mappings, f, indent=2)
            st.success(f"Saved to data/{out_path.name} — see it on the Workspace page.")


# --- Page: Workspace -------------------------------------------------------------

def render_workspace():
    st.header("Workspace")
    st.write("Browse every mapping the Importer has saved.")

    files = list_mapped_files()
    if not files:
        st.info("No mapped files yet — import one on the Importer page first.")
        return

    choice = st.selectbox("Mapped file", files, format_func=lambda p: p.name)
    with open(choice, encoding="utf-8") as f:
        mappings = json.load(f)

    st.dataframe(pd.DataFrame(mappings))


# --- Page: Copilot ---------------------------------------------------------------

def render_copilot():
    st.header("Copilot")
    st.write("Ask questions about your imported data.")

    client = get_client()
    if client is None:
        st.error("Set an Anthropic API key in the sidebar first.")
        return

    # st.session_state persists across reruns (Streamlit re-executes this
    # whole script on every interaction) — without it, the chat history
    # would reset after every message.
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about your data...")
    if prompt:
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Give Claude every saved mapping as context, so it can answer
        # questions like "which columns mapped to Vendor?" or "did any file
        # have low-confidence mappings?".
        context_parts = [
            f"{path.name}:\n{path.read_text(encoding='utf-8')}" for path in list_mapped_files()
        ]
        context = "\n\n".join(context_parts) or "No imported data yet."

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    system=(
                        "You are a helpful copilot for a data-ops platform. Answer "
                        "questions using the imported column mappings below as context.\n\n"
                        + context
                    ),
                    messages=[{"role": "user", "content": prompt}],
                )
            reply = response.content[0].text
            st.markdown(reply)
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})


# --- Entry point -----------------------------------------------------------------

def main():
    st.set_page_config(page_title="CDO AI Platform", layout="wide")
    page = render_sidebar()

    if page == "Importer":
        render_importer()
    elif page == "Workspace":
        render_workspace()
    elif page == "Copilot":
        render_copilot()


if __name__ == "__main__":
    main()
