"""Streamlit case browser for the joint subgraph demo (P5).

Run with:
    .venv/bin/streamlit run demo/app.py

Pre-computed results only — no live LLM. See
`docs/superpowers/specs/2026-05-22-p5-demo-design.md` for the design.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Page config must be the very first Streamlit call (spec §5 risk #1).
st.set_page_config(page_title="DataAgent — Joint Subgraph Demo")
st.title("DataAgent — Joint Subgraph Demo")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo import case_loader  # noqa: E402
from demo.subgraph_render import render_subgraph  # noqa: E402


CURATED_PATH = Path(__file__).parent / "curated_cases.json"


def _load_curated() -> list[dict]:
    if not CURATED_PATH.exists():
        return []
    try:
        data = json.loads(CURATED_PATH.read_text(encoding="utf-8"))
        return data.get("cases", []) or []
    except Exception:
        return []


def _truncate(s: str, n: int = 200) -> str:
    if not s:
        return ""
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def main() -> None:
    curated = _load_curated()

    with st.sidebar:
        st.header("Case Browser")
        chosen_qid: int | None = None
        if curated:
            labels = [f"qid {c['qid']} — {c.get('label', '')}" for c in curated]
            sel = st.selectbox("Curated cases", options=[""] + labels)
            if sel:
                idx = labels.index(sel)
                chosen_qid = int(curated[idx]["qid"])
        else:
            st.caption("(No curated cases yet — use the free-form qid below.)")

        free_qid = st.number_input(
            "Free-form qid", min_value=0, max_value=100000,
            value=0, step=1,
        )
        if st.button("Load qid", use_container_width=True):
            chosen_qid = int(free_qid)

    if chosen_qid is None:
        st.info(
            "Pick a curated case from the sidebar, or enter a qid and click "
            "**Load qid**. The demo browses pre-computed P3/P4 results — no "
            "live LLM call is made."
        )
        return

    try:
        case = case_loader.load_case(chosen_qid)
    except KeyError as e:
        st.error(f"qid {chosen_qid} not found in any result JSON.\n\n{e}")
        return
    except Exception as e:
        st.exception(e)
        return

    # ---- Header block: question + gold SQL ---------------------------------
    st.subheader(f"Question (qid {case['qid']}, db: {case['db_id']}, "
                 f"difficulty: {case['difficulty']})")
    st.write(case["question"])

    st.markdown("**Gold SQL**")
    st.code(case["gold_sql"] or "(none)", language="sql")

    # ---- Joint subgraph ----------------------------------------------------
    st.markdown("---")
    st.subheader("Joint subgraph")
    sg = case["subgraph"]
    if not sg.get("nodes"):
        st.warning("No joint subgraph available for this qid "
                   "(missing graph file or empty retrieval).")
    else:
        fig = render_subgraph(sg["nodes"], sg["edges"])
        st.pyplot(fig)
        st.caption(
            f"{len(sg['nodes'])} nodes / {len(sg['edges'])} edges. "
            "Edges: solid = defines, dashed = depends, dotted = grounds."
        )

    # ---- 4-setting comparison ---------------------------------------------
    st.markdown("---")
    st.subheader("Per-setting comparison")

    order = ["none", "retrieval", "joint", "oracle"]
    cols = st.columns(4)
    for col, name in zip(cols, order):
        entry = case["settings"][name]
        with col:
            st.markdown(f"### {name}")
            if entry.get("missing"):
                st.caption("(no result row for this qid)")
                continue
            correct = bool(entry.get("correct"))
            verdict_color = "#2e7d32" if correct else "#c62828"
            verdict_text = "CORRECT" if correct else "WRONG"
            st.markdown(
                f"<span style='color:{verdict_color};font-weight:bold'>"
                f"{verdict_text}</span>",
                unsafe_allow_html=True,
            )
            st.markdown("**Evidence**")
            ev = _truncate(entry.get("evidence", ""), 200)
            st.write(ev if ev else "_(none)_")
            st.markdown("**Predicted SQL**")
            st.code(entry.get("predicted_sql") or "(none)", language="sql")


if __name__ == "__main__":
    main()
