"""
Recommendations module using DeepSeek-V4-Pro for AI-powered suggestions.
"""

import os
import json
from typing import Optional, Dict, Any

try:
    from huggingface_hub import InferenceClient
except ImportError:
    InferenceClient = None


# DeepSeek-V4-Pro model ID
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V4-Pro"

# Path to questions.json — adjust if your directory structure differs
QUESTIONS_JSON_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "questions.json"
)


def load_questions_db() -> Dict[str, Any]:
    """
    Load questions.json and build a flat lookup dict keyed by question ID.

    questions.json structure:
        {
          "NUMERICAL ABILITY": {
            "medium": [ { "id": "NA-1" | 1, "type": "...", "question": "..." }, ... ],
            "hard":   [ ... ]
          },
          "Applied reasoning": { "medium": [...], "hard": [...] },
          ...
        }

    IDs can be strings ("NA-1", "LR-3") or integers (1, 2, 3 …).
    We store everything under str(id) so lookups are always consistent.
    """
    questions_db = {}
    try:
        with open(QUESTIONS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        for section_key, difficulties in data.items():
            if not isinstance(difficulties, dict):
                continue
            for difficulty, questions in difficulties.items():
                if not isinstance(questions, list):
                    continue
                for q in questions:
                    qid = q.get("id")
                    if qid is None:
                        continue
                    # Normalise: always store under the string version of the id
                    questions_db[str(qid)] = {
                        # Prefer the question's own "type" field; fall back to section key
                        "type": q.get("type", section_key),
                        "question": q.get("question", ""),
                    }
    except Exception as e:
        print(f"Warning: Could not load questions.json: {e}")

    return questions_db


# Load once at module level
_questions_db = load_questions_db()


def get_question_details(qid) -> Dict[str, str]:
    """
    Return {'type': ..., 'question': ...} for the given question ID.
    qid may be an int or a string — both are handled.
    Returns safe defaults when the ID is not found.
    """
    details = _questions_db.get(str(qid))
    if details:
        return details
    return {"type": "Unknown", "question": "Question text not available"}


# ──────────────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_deepseek_client() -> Optional["InferenceClient"]:
    """Initialise and return a Hugging Face InferenceClient, or None."""
    if InferenceClient is None:
        return None
    return InferenceClient()


def format_hsqsns_for_prompt(hsqsns: list) -> str:
    """
    Format a list of high-stress question IDs for insertion into an LLM prompt.

    hsqsns is expected to be a flat list of question IDs, e.g.:
        ["NA-1", 3, "LR-3", 22]

    Each ID is resolved to its type and question text via questions.json.
    """
    if not hsqsns:
        return "No high-stress questions recorded."

    lines = []
    for i, qid in enumerate(hsqsns, 1):
        details = get_question_details(qid)
        qtype    = details["type"]
        question = details["question"]

        lines.append(
            f"{i}. Question ID : {qid}\n"
            f"   Type        : {qtype}\n"
            f"   Content     : {question[:200]}"   # cap length
        )

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Recommendation generators
# ──────────────────────────────────────────────────────────────────────────────

def generate_ai_recommendations(
    hsqsns: list,
    avg_stress: float,
    model: str = DEEPSEEK_MODEL,
) -> str:
    """
    Generate personalised recommendations using DeepSeek-V4-Pro via
    the Hugging Face Inference API.

    Args:
        hsqsns    : flat list of high-stress question IDs (str or int)
        avg_stress: average stress score out of 10
        model     : HF model ID to call

    Returns:
        AI-generated recommendations as a string, or an error message.
    """
    if InferenceClient is None:
        return (
            "Error: huggingface_hub is not installed.\n"
            "Run:  pip install huggingface_hub"
        )

    questions_formatted = format_hsqsns_for_prompt(hsqsns)

    if avg_stress is None:
        avg_stress = 0  # or default safe value

    stress_category = (
        "High"     if avg_stress > 7 else
        "Moderate" if avg_stress > 4 else
        "Low"
    )

    prompt = f"""You are a cognitive assessment expert. \
Based on the following data from a cognitive test session, \
provide personalised recommendations to help the user improve \
their cognitive performance and manage stress.

## User's Stress Data
- Average Stress Level : {avg_stress}/10
- Stress Category      : {stress_category}

## High-Stress Questions
(These are the questions that caused elevated stress during the test.)

{questions_formatted}

## Your Task
Provide the following sections:

### 1. Cognitive Areas to Improve
Analyse which question types (logical, mathematical, verbal, memory, etc.) \
caused the most stress and identify specific areas to work on.

### 2. Practice Recommendations
Suggest concrete exercises or activities targeting each weak area.

### 3. Stress Management Tips
Provide practical techniques to manage test anxiety before and during assessments.

### 4. Overall Assessment
Give a brief summary of the user's cognitive stress patterns.

Format your response with clear headings and bullet points. \
Keep all recommendations practical and immediately actionable."""

    try:
        client = get_deepseek_client()
        if client is None:
            return "Error: Could not initialise Hugging Face InferenceClient."

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7,
        )

        if response and response.choices:
            return response.choices[0].message.content
        return "Error: No response received from the model."

    except Exception as e:
        return f"Error generating AI recommendations: {e}"


def generate_rule_based_recommendations(
    hsqsns: list,
    avg_stress: float,
) -> str:
    """
    Generate recommendations using rule-based logic.
    Used as a fallback when the AI model is unavailable.

    Args:
        hsqsns    : flat list of high-stress question IDs (str or int)
        avg_stress: average stress score out of 10

    Returns:
        Formatted recommendations string.
    """
    if not hsqsns:
        return "No high-stress questions to analyse."

    # ── Count occurrences of each question type ────────────────────────────
    type_counts: Dict[str, int] = {}
    for qid in hsqsns:
        details = get_question_details(qid)
        qtype = details["type"]
        type_counts[qtype] = type_counts.get(qtype, 0) + 1

    # Sort most-frequent first
    sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)

    lines = []

    # ── Header ────────────────────────────────────────────────────────────
    lines.append("## 📊 Cognitive Stress Analysis\n")
    lines.append(f"**Stress Level:** {avg_stress}/10")

    if avg_stress > 7:
        lines.append("**Category:** 🔴 High — Consider dedicated stress management")
    elif avg_stress > 4:
        lines.append("**Category:** 🟡 Moderate — Room for improvement")
    else:
        lines.append("**Category:** 🟢 Low — Good stress management")

    lines.append("\n---\n")

    # ── Areas to improve ──────────────────────────────────────────────────
    lines.append("### 🎯 Areas to Improve\n")
    for qtype, count in sorted_types:
        label = qtype.replace("_", " ").title()
        lines.append(f"- **{label}**: {count} high-stress question(s)")

    lines.append("\n---\n")

    # ── Practice recommendations ──────────────────────────────────────────
    lines.append("### 📝 Practice Recommendations\n")

    seen_recs: set = set()

    def add_rec(key: str, text: str) -> None:
        if key not in seen_recs:
            seen_recs.add(key)
            lines.append(f"- {text}")

    for qtype, _ in sorted_types:
        qt = qtype.lower()

        if any(k in qt for k in ("math", "numerical", "quant", "number")):
            add_rec("math", "**Mathematical**: Practice daily mental arithmetic and number puzzles (e.g. Sudoku, KenKen)")
        if any(k in qt for k in ("logical", "reasoning", "logic")):
            add_rec("logic", "**Logical Reasoning**: Solve logic grid puzzles and pattern-recognition exercises")
        if any(k in qt for k in ("verbal", "language", "vocabulary", "reading")):
            add_rec("verbal", "**Verbal**: Read complex texts daily and build vocabulary using flashcard apps")
        if any(k in qt for k in ("memory", "recall")):
            add_rec("memory", "**Memory**: Play memory-matching games and practise sequence-recall exercises")
        if any(k in qt for k in ("applied", "situational", "judgment")):
            add_rec("applied", "**Applied Reasoning**: Review case-study scenarios and practise situational-judgment tests")

    if not seen_recs:
        lines.append("- Review the question types above and seek targeted practice resources.")

    lines.append("\n---\n")

    # ── Stress management ─────────────────────────────────────────────────
    lines.append("### 🧘 Stress Management Tips\n")

    if avg_stress > 7:
        lines.append("- Practise deep-breathing exercises (4-7-8 technique) before tests")
        lines.append("- Try daily guided meditation (even 10 minutes helps)")
        lines.append("- Take short breaks every 25 minutes during long sessions (Pomodoro method)")
        lines.append("- Ensure 7–8 hours of sleep the night before any assessment")
    elif avg_stress > 4:
        lines.append("- Use progressive muscle relaxation before sitting down to study")
        lines.append("- Break tasks into smaller, time-boxed chunks")
        lines.append("- Stay well hydrated and maintain balanced nutrition")
    else:
        lines.append("- Maintain your current stress-management habits — they're working well")
        lines.append("- Continue regular physical exercise to keep stress low")

    lines.append("\n---\n")

    # ── Overall assessment ────────────────────────────────────────────────
    lines.append("### 📈 Overall Assessment\n")
    most_stressful = sorted_types[0][0].replace("_", " ").title() if sorted_types else "N/A"
    lines.append(f"- Total high-stress questions : {len(hsqsns)}")
    lines.append(f"- Most stressful question type: {most_stressful}")
    lines.append(f"- Average stress level        : {avg_stress}/10")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def get_recommendations(
    hsqsns: list,
    avg_stress: float,
    use_ai: bool = True,
) -> str:
    """
    Return recommendations for the user.

    Args:
        hsqsns    : flat list of high-stress question IDs (str or int)
        avg_stress: average stress score out of 10
        use_ai    : if True, call DeepSeek-V4-Pro; otherwise use rule-based logic

    Returns:
        Formatted recommendations string.
    """
    if use_ai:
        result = generate_ai_recommendations(hsqsns, avg_stress)
        # If AI fails, fall back gracefully
        if result.startswith("Error"):
            print(f"AI recommendations failed: {result}\nFalling back to rule-based.")
            return generate_rule_based_recommendations(hsqsns, avg_stress)
        return result
    return generate_rule_based_recommendations(hsqsns, avg_stress)