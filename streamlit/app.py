"""
Human Evaluation App
====================
Displays conversation prompts one at a time and asks the annotator whether
the target speaker should SPEAK or remain SILENT.

Run:
    cd streamlit/
    streamlit run app.py
"""

import json
from collections import Counter
from pathlib import Path

import streamlit as st

from sample_questions import OUTPUT_PATH, sample_questions

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
SAMPLED_PATH = OUTPUT_PATH           # streamlit/sampled_100.jsonl
RESULTS_PATH = BASE_DIR / "human_eval_results.jsonl"

# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------
DATASET_COLORS = {
    "AMI": "#1f77b4",       # blue
    "Friends": "#2ca02c",   # green
    "SPGI": "#ff7f0e",      # orange
}

CATEGORY_LABELS = {
    "SPEAK_explicit": "SPEAK — explicit address",
    "SPEAK_implicit": "SPEAK — implicit",
    "SILENT_no_ref": "SILENT — no reference",
    "SILENT_ref": "SILENT — referenced",
}


def dataset_badge(name: str) -> str:
    color = DATASET_COLORS.get(name, "#888")
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:0.8rem;font-weight:600;">{name}</span>'
    )


def category_badge(cat: str) -> str:
    label = CATEGORY_LABELS.get(cat, cat)
    return (
        f'<span style="background:#e0e0e0;color:#333;padding:3px 10px;'
        f'border-radius:12px;font-size:0.8rem;">{label}</span>'
    )


# ---------------------------------------------------------------------------
# Data loading / saving
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Preparing questions…")
def load_questions() -> list[dict]:
    if not SAMPLED_PATH.exists():
        sample_questions()
    questions = []
    with open(SAMPLED_PATH) as fh:
        for line in fh:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def load_existing_results() -> dict[str, str]:
    """Return mapping decision_point_id → human_judgement from saved results."""
    answers: dict[str, str] = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if "decision_point_id" in rec and "human_judgement" in rec:
                        answers[rec["decision_point_id"]] = rec["human_judgement"]
                except json.JSONDecodeError:
                    pass
    return answers


def save_all_results(questions: list[dict], answers: dict[str, str]) -> None:
    """Rewrite the results file with all current answers."""
    with open(RESULTS_PATH, "w") as fh:
        for q in questions:
            qid = q["decision_point_id"]
            if qid in answers:
                record = {**q, "human_judgement": answers[qid]}
                fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def init_state() -> None:
    if "questions" not in st.session_state:
        st.session_state.questions = load_questions()

    if "answers" not in st.session_state:
        st.session_state.answers = load_existing_results()

    if "current_idx" not in st.session_state:
        # Jump to first unanswered question
        answered_ids = set(st.session_state.answers.keys())
        first_unanswered = next(
            (
                i
                for i, q in enumerate(st.session_state.questions)
                if q["decision_point_id"] not in answered_ids
            ),
            len(st.session_state.questions),  # all done
        )
        st.session_state.current_idx = first_unanswered


# ---------------------------------------------------------------------------
# Answer recording
# ---------------------------------------------------------------------------
def record_answer(judgement: str) -> None:
    questions = st.session_state.questions
    idx = st.session_state.current_idx
    q = questions[idx]
    st.session_state.answers[q["decision_point_id"]] = judgement
    save_all_results(questions, st.session_state.answers)
    # Advance to next question
    if idx + 1 < len(questions):
        st.session_state.current_idx = idx + 1


# ---------------------------------------------------------------------------
# Completion screen
# ---------------------------------------------------------------------------
def show_completion() -> None:
    st.balloons()
    st.success("All 100 questions answered!")

    questions = st.session_state.questions
    answers = st.session_state.answers

    total = len(answers)
    speak_count = sum(1 for v in answers.values() if v == "SPEAK")
    silent_count = total - speak_count

    st.markdown("## Results Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Answered", total)
    col2.metric("SPEAK", speak_count)
    col3.metric("SILENT", silent_count)

    st.markdown("### By Dataset")
    by_dataset: dict[str, Counter] = {}
    for q in questions:
        qid = q["decision_point_id"]
        if qid not in answers:
            continue
        ds = q.get("dataset", "Unknown")
        if ds not in by_dataset:
            by_dataset[ds] = Counter()
        by_dataset[ds][answers[qid]] += 1

    cols = st.columns(len(by_dataset))
    for col, (ds, cnt) in zip(cols, by_dataset.items()):
        col.markdown(f"**{ds}**")
        col.write(f"SPEAK: {cnt['SPEAK']}")
        col.write(f"SILENT: {cnt['SILENT']}")

    st.markdown("### By Category")
    by_cat: dict[str, Counter] = {}
    for q in questions:
        qid = q["decision_point_id"]
        if qid not in answers:
            continue
        cat = q.get("category", "Unknown")
        if cat not in by_cat:
            by_cat[cat] = Counter()
        by_cat[cat][answers[qid]] += 1

    for cat, cnt in by_cat.items():
        st.write(f"**{CATEGORY_LABELS.get(cat, cat)}** — SPEAK: {cnt['SPEAK']}, SILENT: {cnt['SILENT']}")

    # Download button
    lines = []
    for q in questions:
        qid = q["decision_point_id"]
        if qid in answers:
            lines.append(json.dumps({**q, "human_judgement": answers[qid]}))
    st.download_button(
        label="Download Results (JSONL)",
        data="\n".join(lines) + "\n",
        file_name="human_eval_results.jsonl",
        mime="application/jsonl",
    )

    if st.button("Review answers from the beginning"):
        st.session_state.current_idx = 0
        st.rerun()


# ---------------------------------------------------------------------------
# Main question page
# ---------------------------------------------------------------------------
def show_question(idx: int) -> None:
    questions = st.session_state.questions
    answers = st.session_state.answers
    total = len(questions)
    q = questions[idx]

    qid = q["decision_point_id"]
    existing_answer = answers.get(qid)

    # --- Progress ---
    answered_count = len(answers)
    st.progress(answered_count / total, text=f"{answered_count} / {total} answered")

    # --- Header row ---
    st.markdown(f"### Question {idx + 1} &nbsp;/&nbsp; {total}")

    st.divider()

    # --- Target speaker callout ---
    target = q.get("target_speaker", "?")

    st.subheader(f"Should speaker `{target}` speak next?")

    all_speakers = q.get("all_speakers", [])
    st.caption(f"All participants: {', '.join(all_speakers)}")

    # --- Conversation context ---
    context_turns = q.get("context_turns", [])
    if context_turns:
        st.markdown("**Conversation Context**")
        with st.container(border=True):
            for turn in context_turns:
                spk = turn.get("speaker", "?")
                txt = turn.get("text", "")
                is_target = spk == target
                # Bold + italic label for the target speaker so it stands out
                # without relying on colour (works in both light and dark mode)
                if is_target:
                    label = f"**[{spk}]** (target)"
                else:
                    label = f"[{spk}]"
                st.markdown(f"{label}&nbsp; {txt}")
    else:
        st.info("No prior context available for this sample.")

    # --- Current turn ---
    current = q.get("current_turn", {})
    cur_spk = current.get("speaker", "?")
    cur_txt = current.get("text", "")
    st.markdown("**Current Turn** *(just said)*")
    # Use st.info so the box inherits the theme colours properly
    st.info(f"**[{cur_spk}]** {cur_txt}")

    st.markdown("")  # spacer

    # --- Judgement buttons ---
    st.markdown(f"**Given the above, should `{target}` speak next?**")

    btn_col1, btn_col2, _ = st.columns([1, 1, 2])

    speak_type = "primary" if existing_answer == "SPEAK" else "secondary"
    silent_type = "primary" if existing_answer == "SILENT" else "secondary"

    with btn_col1:
        if st.button(
            "🗣️ SPEAK",
            use_container_width=True,
            type=speak_type,
            key=f"speak_{idx}",
        ):
            record_answer("SPEAK")
            st.rerun()

    with btn_col2:
        if st.button(
            "🤫 SILENT",
            use_container_width=True,
            type=silent_type,
            key=f"silent_{idx}",
        ):
            record_answer("SILENT")
            st.rerun()

    if existing_answer:
        st.success(f"Your answer: **{existing_answer}**")

    st.divider()

    # --- Navigation ---
    nav_left, nav_right = st.columns([1, 1])

    with nav_left:
        if idx > 0:
            if st.button("← Back", use_container_width=True):
                st.session_state.current_idx -= 1
                st.rerun()

    with nav_right:
        can_advance = existing_answer is not None and idx + 1 < total
        if can_advance:
            if st.button("Next →", use_container_width=True, type="primary"):
                st.session_state.current_idx += 1
                st.rerun()
        elif existing_answer and idx + 1 == total:
            if st.button("See Results →", use_container_width=True, type="primary"):
                st.session_state.current_idx = total
                st.rerun()


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Human Eval",
    page_icon="🗣️",
    layout="centered",
)

st.title("Human Evaluation")
st.caption(
    "For each conversation snippet, decide whether the highlighted speaker "
    "should speak next (SPEAK) or stay silent (SILENT)."
)

init_state()

total_questions = len(st.session_state.questions)
idx = st.session_state.current_idx

if idx >= total_questions:
    show_completion()
else:
    show_question(idx)
