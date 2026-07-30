"""Agent loop implementing a tool-calling workflow for CSV/XLSX header mapping."""

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

from pii import scrub_pii

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Ordered tuple used in the system prompt and for canonical_field normalisation.
CANONICAL_FIELDS = ("Customer", "Job", "Invoice", "Payment", "Task", "Vendor", "VendorID")
_CANONICAL_LOWER: dict[str, str] = {f.lower(): f for f in CANONICAL_FIELDS}


class AgentLoopError(RuntimeError):
    """Raised when the agent loop exits without producing a final response."""

SYSTEM_PROMPT = """You are a data mapper. Use the available tools to load CSV/XLSX headers,
inspect individual columns, and save a final JSON mapping.

Canonical fields: Customer, Job, Invoice, Payment, Task, Vendor, VendorID

Each mapping object passed to save_mappings MUST include:
- source_column: the original column name
- canonical_field: the matched canonical field, or null if unmatched
- confidence: a float between 0 and 1 representing your certainty
- reasoning: a brief explanation of why this mapping was chosen

For columns that do not match any canonical field, use null as the canonical_field value
and add a "notes" key explaining why it was not mapped.

Workflow:
1. Call load_headers() once to get all columns.
   - If the result includes "sheet_names", check whether the active_sheet is the correct
     data sheet. Re-call load_headers with the right sheet_name if needed.
   - If the result includes "duplicate_headers", note which columns were renamed.
2. Call inspect_column() for each column. You may inspect multiple columns per turn.
3. After all columns are mapped, call save_mappings(). Check that entries_written in
   the response equals the number of headers from load_headers. If it is less, map the
   remaining columns and call save_mappings again with the complete list.

Only reason about mapping through the provided tools; do not act on data directly."""

TOOLS = [
    {
        "name": "load_headers",
        "description": (
            "Load column headers and sample values from a CSV or XLSX file. "
            "Returns the first 3 non-null sample values per column. "
            "For XLSX files with multiple sheets, also returns sheet_names and active_sheet."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the input file to read",
                },
                "sheet_name": {
                    "type": "string",
                    "description": (
                        "XLSX only: name or 0-based index of the sheet to read. "
                        "Omit to use the first sheet."
                    ),
                },
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "inspect_column",
        "description": (
            "Return a single column header and its sample values so the agent can "
            "perform the mapping reasoning in the next turn."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "header": {
                    "type": "string",
                    "description": "The column header name",
                },
                "sample_values": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Example values for the column",
                },
            },
            "required": ["header", "sample_values"],
        },
    },
    {
        "name": "save_mappings",
        "description": "Save the mapping results to a JSON file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to save the JSON mappings",
                },
                "mappings": {
                    "type": "array",
                    "description": "List of mapping objects",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_column": {
                                "type": "string",
                                "description": "Original column name from the source file",
                            },
                            "canonical_field": {
                                "description": "Matched canonical field name, or null if unmatched",
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence score between 0 and 1",
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation of why this mapping was chosen",
                            },
                        },
                        "required": ["source_column", "canonical_field", "confidence", "reasoning"],
                    },
                },
            },
            "required": ["filepath", "mappings"],
        },
    },
]


def _resolve_filepath(filepath: str) -> Path:
    """Resolve a file path relative to cwd, the script directory, or the repo root."""
    candidate = Path(filepath).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    for base in (script_dir, repo_root):
        resolved = base / filepath
        if resolved.exists():
            return resolved

    return candidate


# Encodings tried in order; utf-8-sig strips the BOM that Excel adds to UTF-8 exports.
_CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

# Pandas' default NA strings with "nan"/"NaN" removed so a column literally named
# "nan" is kept as the string "nan" rather than silently converted to float NaN.
_CSV_NA_VALUES = frozenset({
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN",
    "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>",
    "N/A", "NA", "NULL", "NaN", "None", "n/a", "null",
})


def _read_raw_csv(path: Path, nrows: int) -> "pd.DataFrame":
    """
    Read a CSV with explicit delimiter probing and encoding fallback.
    Tries comma, tab, semicolon, and pipe in order before falling back to a
    single-column read. Uses Python's csv.reader to pre-count the maximum column
    width across all rows so that pandas can NaN-pad shorter rows (e.g. metadata
    rows that precede the real headers) instead of erroring on mismatched widths.
    Returns a header=None DataFrame.
    """
    import csv as csvmod
    import pandas as pd

    last_err: Exception | None = None
    for enc in _CSV_ENCODINGS:
        try:
            for sep in (",", "\t", ";", "|"):
                # Count the widest row using Python's csv.reader (handles quoting).
                max_cols = 0
                try:
                    with open(path, encoding=enc, newline="") as fh:
                        for i, row in enumerate(csvmod.reader(fh, delimiter=sep)):
                            if i >= nrows:
                                break
                            max_cols = max(max_cols, len(row))
                except UnicodeDecodeError:
                    raise
                except Exception:
                    continue

                if max_cols < 2:
                    continue

                # Providing names= tells pandas exactly how many columns to expect,
                # so rows narrower than max_cols are NaN-padded rather than rejected.
                # keep_default_na=False + na_values preserves the string "nan" as a
                # column name instead of silently converting it to float NaN.
                return pd.read_csv(
                    path,
                    header=None,
                    names=list(range(max_cols)),
                    nrows=nrows,
                    encoding=enc,
                    sep=sep,
                    na_values=_CSV_NA_VALUES,
                    keep_default_na=False,
                )

            # Fallback: single-column file or exotic delimiter
            return pd.read_csv(
                path, header=None, nrows=nrows, encoding=enc,
                na_values=_CSV_NA_VALUES, keep_default_na=False,
            )

        except UnicodeDecodeError as e:
            last_err = e
    raise last_err  # type: ignore[misc]


def _find_header_row(df_raw: "pd.DataFrame") -> int:
    """
    Scan the first few rows of a header=None DataFrame to locate the real header row.

    Primary signal: first row where >=50% of non-null cells are strings (catches text
    headers and skips numeric-only data rows).

    Fallback: if no string-dominated row is found (e.g. all-numeric column names such
    as year ranges), return the first non-sparse row — i.e. the first row that has at
    least 2 non-null cells, which is the row that follows any single-cell metadata rows.
    """
    first_non_sparse: int | None = None
    for i in range(min(5, len(df_raw))):
        row = df_raw.iloc[i].dropna()
        if len(row) < 2:
            continue
        if sum(1 for v in row if isinstance(v, str)) / len(row) >= 0.5:
            return i
        if first_non_sparse is None:
            first_non_sparse = i
    return first_non_sparse if first_non_sparse is not None else 0


def _header_str(v: Any) -> str:
    """
    Convert a single header-row cell to a clean string.
    - None / float NaN → "" (absent or merged cell)
    - Whole-number float → integer string (2023.0 → "2023")
    - str → returned unchanged, including the literal text "nan"
    - Other non-float scalars (numpy types, pd.NA) → str(), mapping "<NA>" to ""
    """
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:  # NaN check (NaN is the only value not equal to itself)
            return ""
        try:
            i = int(v)
            return str(i) if v == i else str(v)
        except (OverflowError, ValueError):
            return str(v)
    if isinstance(v, str):
        return v  # preserve as-is, including the literal string "nan"
    # Handles numpy integers, booleans, pd.NA, etc.
    s = str(v)
    return "" if s.lower() in ("nan", "<na>") else s


def _df_from_raw(
    df_raw: "pd.DataFrame", header_row: int
) -> tuple["pd.DataFrame", list[str]]:
    """
    Build a DataFrame from a raw (header=None) DataFrame.

    Returns (df, duplicate_originals) where duplicate_originals is the list of
    column names that appeared more than once in the source header row.
    Duplicate column names are suffixed with .1, .2, … to avoid collisions.
    """
    raw = [_header_str(v) for v in df_raw.iloc[header_row].tolist()]

    counts = Counter(raw)
    duplicates = [h for h, c in counts.items() if c > 1 and h]

    seen: dict[str, int] = {}
    headers: list[str] = []
    for h in raw:
        n = seen.get(h, 0)
        headers.append(f"{h}.{n}" if n > 0 else h)
        seen[h] = n + 1

    df = df_raw.iloc[header_row + 1 :].reset_index(drop=True)
    df.columns = headers
    return df, duplicates


def load_headers(filepath: str, sheet_name: str | None = None) -> dict[str, Any]:
    """Read a CSV or Excel file and return column headers plus sample values."""
    try:
        import pandas as pd

        resolved_path = _resolve_filepath(filepath)
        if not resolved_path.exists():
            return {
                "error": f"File not found: {filepath}. Tried: {resolved_path}",
                "success": False,
            }

        result_extra: dict[str, Any] = {}
        is_excel = resolved_path.suffix.lower() in (".xlsx", ".xls", ".xlsm")

        if is_excel:
            xl = pd.ExcelFile(resolved_path)
            all_sheets = xl.sheet_names
            target = sheet_name if sheet_name is not None else all_sheets[0]
            if len(all_sheets) > 1:
                result_extra["sheet_names"] = all_sheets
                result_extra["active_sheet"] = target

            df_raw = pd.read_excel(
                resolved_path, sheet_name=target, header=None, nrows=20
            )
        else:
            df_raw = _read_raw_csv(resolved_path, nrows=20)

        if len(df_raw) == 0:
            return {"error": "File contains no readable rows.", "success": False}

        header_row = _find_header_row(df_raw)
        df, duplicates = _df_from_raw(df_raw, header_row)

        headers = df.columns.tolist()
        samples = {col: df[col].dropna().astype(str).tolist()[:3] for col in headers}

        result: dict[str, Any] = {
            "headers": headers,
            "samples": samples,
            "success": True,
            **result_extra,
        }
        if duplicates:
            result["duplicate_headers"] = duplicates
        return result

    except Exception as e:
        return {"error": str(e), "success": False}


def inspect_column(header: str, sample_values: list[str]) -> dict[str, Any]:
    """Return a column header and its sample values for agent reasoning."""
    return {
        "header": header,
        "sample_values": sample_values,
        "success": True,
    }


def save_mappings(filepath: str, mappings: list) -> dict[str, Any]:
    """
    Write the mapping results to a JSON file, creating parent directories as needed.
    Normalises canonical_field casing (e.g. "customer" → "Customer") and returns a
    warning for any values not in CANONICAL_FIELDS.
    """
    try:
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        unknown: list[str] = []
        normalised: list[Any] = []
        for entry in mappings:
            if isinstance(entry, dict):
                entry = dict(entry)  # shallow copy — don't mutate the caller's list
                cf = entry.get("canonical_field")
                if cf is not None:
                    canonical = _CANONICAL_LOWER.get(str(cf).lower())
                    if canonical is not None:
                        entry["canonical_field"] = canonical
                    else:
                        unknown.append(str(cf))
            normalised.append(entry)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(normalised, f, indent=2)

        result: dict[str, Any] = {
            "success": True,
            "message": f"Mappings saved to {filepath}",
            "entries_written": len(normalised),
        }
        if unknown:
            result["warnings"] = [f"Unrecognised canonical_field values (not in CANONICAL_FIELDS): {unknown}"]
        return result
    except Exception as e:
        return {"error": str(e), "success": False}


def process_tool_call(tool_name: str, tool_input: dict) -> str:
    """Dispatch tool calls from the agent to the matching Python helper."""
    if tool_name == "load_headers":
        result = load_headers(tool_input["filepath"], tool_input.get("sheet_name"))
    elif tool_name == "inspect_column":
        result = inspect_column(tool_input["header"], tool_input["sample_values"])
    elif tool_name == "save_mappings":
        result = save_mappings(tool_input["filepath"], tool_input["mappings"])
    else:
        result = {"error": f"Unknown tool: {tool_name}"}

    return json.dumps(result)


def agent_loop(user_message: str, max_iterations: int = 50) -> str:
    """
    Run the agentic loop with tool calling.

    The agent:
    1. Receives a user message
    2. Calls Claude with available tools
    3. If Claude returns tool calls, executes them
    4. Feeds results back to Claude
    5. Repeats until Claude provides a final response (no more tool calls)

    Args:
        user_message: The initial user request
        max_iterations: Max number of loop iterations to prevent infinite loops

    Returns:
        The final text response from Claude
    """
    messages = [{"role": "user", "content": user_message}]
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n[Iteration {iteration}] Calling Claude...")

        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        tool_calls_made = False
        tool_results = []

        for block in response.content:
            if block.type == "tool_use":
                tool_calls_made = True
                tool_name = block.name
                tool_input = block.input
                tool_use_id = block.id

                print(f"  → Tool call: {tool_name}")
                # Scrub before printing: tool_input carries raw sample values
                # straight out of the uploaded file (inspect_column calls),
                # and these prints land in the server's process logs.
                print(f"    Input: {scrub_pii(json.dumps(tool_input, indent=2))}")

                tool_result = process_tool_call(tool_name, tool_input)
                print(f"    Result: {scrub_pii(tool_result)}")

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": tool_result,
                    }
                )

        if tool_results:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        if not tool_calls_made:
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

    raise AgentLoopError(
        f"Agent loop did not produce a final response within {max_iterations} iterations."
    )


def main():
    """Demo entry point: run the agent loop with a sample request."""
    user_request = (
        "Load the file at BOSS/sample.csv, inspect each column with available tools, "
        "and save the mapping results to BOSS/sample.mapped.json"
    )

    print("=" * 70)
    print("AGENT LOOP DEMO: CSV Header Mapping")
    print("=" * 70)
    print(f"\nUser Request: {user_request}\n")

    try:
        final_response = agent_loop(user_request)
    except AgentLoopError as e:
        print(f"\nERROR: {e}")
        return

    print("\n" + "=" * 70)
    print("FINAL RESPONSE:")
    print("=" * 70)
    print(final_response)


if __name__ == "__main__":
    main()
