"""Planner + critic for the auto-evolve loop.

Generates and critiques multi-phase implementation plans using a manager
agent (with swarm/megaswarm tools). Both functions are designed to be
robust — they return [] rather than crashing on malformed model output.
"""

from __future__ import annotations

import re

from .autopilot import parse_phases

# Regex to extract a justification line from a phase body.
# Matches: "Justification: ..." at the start of a line (case-insensitive,
# optional whitespace after colon).
_JUSTIFICATION_RE = re.compile(r"^Justification:\s*(.*)$", re.MULTILINE | re.IGNORECASE)


def generate_plan(manager, context: dict) -> list[dict]:
    """Ask the manager to propose up to 5 improvement phases for the repo.

    *manager* is an Agent instance (with swarm/megaswarm tools).
    *context* is a dict with optional string keys such as ``'outline'``,
    ``'tree'``, ``'readme'``, ``'summary'`` — each describing the current
    repo state.

    Returns a list of phase dicts (``{'title': ..., 'body': ...}``) parsed
    via ``autopilot.parse_phases``, or ``[]`` if the model returned no
    valid phases or an error occurred.
    """
    # Build a prompt that includes every available context snippet.
    context_blocks: list[str] = []
    for key, value in context.items():
        if value:  # skip empty/None values
            context_blocks.append(f"=== {key} ===\n{value}")
    context_section = "\n\n".join(context_blocks)

    prompt = f"""\
You are a planning specialist. Analyse the current state of the repo
described below and propose up to 5 phases of improvements.

{context_section}

=== INSTRUCTIONS ===
1. Propose AT MOST 5 phases. Fewer is fine; quality over quantity.
2. Each phase MUST improve correctness, usefulness, or safety of the
   project. Do NOT propose:
   - Pure refactors (no functional change)
   - Speculative features (vague, no clear benefit)
   - Cosmetic changes (whitespace, formatting, naming)
3. Output each phase as a level-2 heading followed by a clear description
   of what to implement and why it matters. Use EXACTLY this format:

## Phase N — Short Title
Detailed description of the change. Include:
- What file(s) to touch
- What the change should accomplish
- Why this improves correctness / usefulness / safety

Stay focused and concrete. Do NOT include any preamble or commentary
outside the phase headings — your entire output must be parseable."""

    try:
        raw = manager.run(prompt)
        if raw is None:
            raw = ""
        phases = parse_phases(raw)
        return phases
    except Exception:
        return []


def critique_plan(manager, phases: list[dict]) -> list[dict]:
    """Critique a list of proposed phases and drop ones that shouldn't run.

    *manager* is an Agent instance.
    *phases* is a list of phase dicts (``{'title': ..., 'body': ...}``).

    Returns a (possibly shorter) list of phase dicts, now with an extra
    ``'justification'`` key, containing only the phases that pass the
    critic's bar. Returns ``[]`` on malformed output or if all phases are
    rejected.
    """
    if not phases:
        return []

    # Build the list of phases for the prompt.
    phase_text_blocks: list[str] = []
    for i, ph in enumerate(phases, 1):
        phase_text_blocks.append(
            f"### Proposed Phase {i}\n"
            f"**Title:** {ph['title']}\n"
            f"**Body:** {ph['body']}"
        )
    phase_text = "\n\n".join(phase_text_blocks)

    prompt = f"""\
You are a strict code-review critic. Below are {len(phases)} proposed
improvement phases for a software project. For each phase, score it (1-10)
and decide whether to KEEP or DROP it.

=== CRITERIA ===
DROP any phase that is:
- A pure refactor (no functional change)
- A speculative feature (vague, no clear benefit)
- Not clearly improving correctness, usefulness, or safety
- Cosmetic only (formatting, naming, whitespace)

KEEP phases that concretely improve the project.

=== REQUIRED OUTPUT FORMAT ===
For each phase you KEEP, output:

## Phase N — Title
Justification: A brief explanation of why this phase is kept and what score (1-10) it earned.
[original body text]

Output ONLY the surviving phases — do NOT include dropped phases at all.
If you drop ALL phases, output the single word: NONE

=== PROPOSED PHASES ===
{phase_text}"""

    try:
        raw = manager.run(prompt)
        if raw is None:
            raw = ""
    except Exception:
        return []

    raw_stripped = raw.strip()

    # If the model says NONE (or returns empty), there are no survivors.
    if not raw_stripped or raw_stripped.upper() == "NONE":
        return []

    # Parse the surviving phases using parse_phases, then extract
    # justifications from each body.
    try:
        surviving = parse_phases(raw)
    except Exception:
        return []

    result: list[dict] = []
    for ph in surviving:
        body = ph["body"]
        just_match = _JUSTIFICATION_RE.search(body)
        if just_match:
            justification = just_match.group(1).strip()
            # Remove the justification line(s) from the body to keep it clean.
            # Also remove any blank lines left behind.
            clean_body = _JUSTIFICATION_RE.sub("", body).strip()
        else:
            justification = ""
            clean_body = body.strip()

        result.append({
            "title": ph["title"],
            "body": clean_body,
            "justification": justification,
        })

    return result
