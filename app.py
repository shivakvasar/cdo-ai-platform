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

import mapper_agent
from pii import scrub_pii

# --- Constants ---------------------------------------------------------------
MODEL = "claude-sonnet-4-5"
DATA_DIR = Path(__file__).parent / "data"  # confirmed mappings the Workspace reads
UPLOAD_DIR = Path(__file__).parent / "uploads"  # raw files, so mapper_agent has a filepath to read
STAGING_DIR = DATA_DIR / "_staging"  # mapper_agent output pending user Confirm; a
# subdir of DATA_DIR so list_mapped_files()'s non-recursive glob skips it automatically

# --- Shared helpers ------------------------------------------------------------

def get_api_key() -> str | None:
    """The Anthropic API key, wherever it comes from.

    Checks the ANTHROPIC_API_KEY environment variable first (e.g. set via a
    .env-loaded shell); falls back to the sidebar text input in
    render_sidebar() so the app still works the very first time, before
    anyone has set up a .env file. Returns None when no key is available yet,
    so callers can show a friendly message instead of crashing the page.
    """
    return os.environ.get("ANTHROPIC_API_KEY") or st.session_state.get("api_key") or None


def get_client() -> anthropic.Anthropic | None:
    """Build an Anthropic client from get_api_key(), or None if no key is set yet.

    Streamlit reruns this whole script top-to-bottom every time the user
    interacts with a widget, so this needs to be cheap to call repeatedly.
    """
    api_key = get_api_key()
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


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
    st.write("Upload a CSV or XLSX file and map its columns to canonical fields using mapper_agent.")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    if uploaded_file is None:
        return

    df = load_dataframe(uploaded_file)
    st.subheader("Preview")
    st.dataframe(df.head())

    stem = Path(uploaded_file.name).stem

    if st.button("Run mapper_agent"):
        api_key = get_api_key()
        if api_key is None:
            st.error("Set an Anthropic API key in the sidebar first.")
            return

        # mapper_agent works off filepaths (it re-reads the file itself via its
        # own load_headers tool), so the in-memory upload needs to land on disk
        # first. mapper_agent.client is built at import time from whatever
        # ANTHROPIC_API_KEY happened to be set then, which may predate a key
        # typed into the sidebar — rebuild it here so it's never stale.
        UPLOAD_DIR.mkdir(exist_ok=True)
        upload_path = UPLOAD_DIR / uploaded_file.name
        upload_path.write_bytes(uploaded_file.getvalue())

        os.environ["ANTHROPIC_API_KEY"] = api_key
        mapper_agent.client = anthropic.Anthropic(api_key=api_key)

        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        staging_path = STAGING_DIR / f"{stem}.mapped.json"
        prompt = (
            f"Load the file at {upload_path}, inspect each column with the "
            f"available tools, map each to the best canonical field, and save "
            f"the mapping results to {staging_path}."
        )

        with st.spinner("Running mapper_agent..."):
            try:
                mapper_agent.agent_loop(prompt)
            except mapper_agent.AgentLoopError as e:
                st.error(f"mapper_agent did not finish: {e}")
                return

        if not staging_path.exists():
            st.error("mapper_agent finished without saving a mapping file.")
            return

        with open(staging_path, encoding="utf-8") as f:
            mappings = json.load(f)

        # Stashed in session_state (rather than a local variable) so the
        # mapping table below survives the rerun that happens after this
        # button click and stays visible until a *different* file is picked.
        st.session_state["last_mappings"] = mappings
        st.session_state["last_filename"] = uploaded_file.name

    mappings = st.session_state.get("last_mappings")
    if mappings and st.session_state.get("last_filename") == uploaded_file.name:
        st.subheader("Mapping results")
        st.dataframe(pd.DataFrame(mappings))

        if st.button("Confirm"):
            DATA_DIR.mkdir(exist_ok=True)
            out_path = DATA_DIR / f"{stem}.mapped.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(mappings, f, indent=2)
            st.success(f"Saved to data/{out_path.name} — see it on the Workspace page.")


# --- Page: Workspace -------------------------------------------------------------

WORKSPACE_TABS = {"Customers": "Customer", "Jobs": "Job", "Invoices": "Invoice"}


def render_workspace():
    st.header("Workspace")
    st.write("Browse confirmed mappings, grouped by canonical field.")

    files = list_mapped_files()
    if not files:
        st.info("No mapped files yet — import one on the Importer page first.")
        return

    # Pool every confirmed mapping entry across all files, tagged with its
    # source filename, so each tab below can filter down to its own field.
    rows = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            mappings = json.load(f)
        for entry in mappings:
            if isinstance(entry, dict):
                rows.append({"file": path.name, **entry})
    all_mappings = pd.DataFrame(rows)

    for tab, field in zip(st.tabs(list(WORKSPACE_TABS)), WORKSPACE_TABS.values()):
        with tab:
            subset = all_mappings[all_mappings.get("canonical_field") == field]
            if subset.empty:
                st.info(f"No columns mapped to {field} yet.")
            else:
                st.dataframe(subset)


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
        # Mapped files store a few real sample_values per column (e.g. to show
        # what "Customer" looked like in the source file), so this context can
        # contain emails/phone numbers straight from someone's spreadsheet.
        # scrub_pii masks those before this ever leaves the machine as part of
        # the Claude API call below.
        context = scrub_pii("\n\n".join(context_parts)) or "No imported data yet."

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
