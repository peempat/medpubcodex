"""Deterministic scope layer for the Health Information Assistant demo.

This module is a small rule-based filter that runs *outside* the language model.
It exists so that the demo does not depend on prompt adherence alone for the two
things the project must not do: assert a diagnosis, and hand out personalised
medication dosing.

This is **not** a clinically validated safety system. It is a simple keyword and
pattern filter written for a research demo. It will miss phrasings it was not
written for, and it will occasionally redirect a question that was harmless.
Do not present it as a medical safeguard.

The layer runs twice per turn:

``check_request``
    Before generation. Classifies the user question and decides whether to run
    the model at all, and what extra instruction to add to the system prompt.

``review_answer``
    After generation. Looks for direct diagnostic assertions in the model output
    and attaches a correction notice when one is found.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

DISCLAIMER = (
    "**For research and educational purposes only.** This application does not "
    "provide medical diagnosis and does not replace a qualified healthcare "
    "professional."
)

EMERGENCY_NOTICE = (
    "### Please seek medical care now\n\n"
    "The symptoms you described can be associated with conditions that need "
    "urgent assessment. This demo cannot evaluate them and will not try.\n\n"
    "- If you are in immediate danger, call your local emergency number "
    "(Thailand: **1669**; EU: **112**; US: **911**).\n"
    "- Otherwise contact a doctor, an emergency department, or a local "
    "healthcare service as soon as you can.\n\n"
    "A person with the right training needs to assess this, not a research demo."
)

CRISIS_NOTICE = (
    "### Please talk to someone who can help\n\n"
    "It sounds like you may be going through something very difficult. This "
    "research demo is not able to support you with this, and it should not try.\n\n"
    "- Thailand: Department of Mental Health hotline **1323** (24 hours)\n"
    "- International directory: <https://findahelpline.com>\n"
    "- If you are in immediate danger, call your local emergency number.\n\n"
    "Talking to a trained person is the right next step here."
)


# --------------------------------------------------------------------------- #
# patterns
# --------------------------------------------------------------------------- #
# Symptom clusters that should never be routed through a general-information
# model. Kept deliberately narrow so ordinary questions are not swept up.
_EMERGENCY_PATTERNS = (
    r"\bchest pain\b",
    r"\bcrushing (?:chest )?pain\b",
    r"\b(?:can(?:no|')?t|cannot|difficulty|trouble) breath",
    r"\bshortness of breath\b.{0,40}\b(?:sudden|severe|now)\b",
    r"\bstroke symptoms?\b",
    r"\b(?:face|facial) droop",
    r"\bslurred speech\b",
    r"\bsudden (?:weakness|numbness) (?:on )?one side\b",
    r"\bcoughing (?:up )?blood\b",
    r"\bvomiting blood\b",
    r"\bsevere bleeding\b",
    r"\bbleeding (?:that )?(?:won(?:no|')?t|will not) stop\b",
    r"\banaphyla",
    r"\bthroat (?:is )?closing\b",
    r"\bunconscious\b",
    r"\bnot (?:breathing|responding|waking up)\b",
    r"\bseizure\b.{0,30}\b(?:right now|ongoing|still)\b",
    r"\boverdose\b",
    r"\bpoison(?:ed|ing)\b",
    r"เจ็บ(?:หน้า)?อก",
    r"หายใจไม่ออก",
    r"หมดสติ",
    r"ชักเกร็ง",
    r"เลือดออกไม่หยุด",
    r"อาเจียนเป็นเลือด",
    r"ปากเบี้ยว",
)

_CRISIS_PATTERNS = (
    r"\b(?:kill|hurt|harm) (?:my ?self|myself)\b",
    r"\bsuicid",
    r"\bend my life\b",
    r"\bwant to die\b",
    r"\bself[- ]harm\b",
    r"ฆ่าตัวตาย",
    r"ทำร้ายตัวเอง",
    r"อยากตาย",
)

# Requests for a personal diagnosis. These are redirected to general information,
# never refused outright - the user still gets educational content.
_DIAGNOSIS_PATTERNS = (
    r"\bwhat (?:disease|illness|condition|sickness)\s+(?:do|have)?\s*i\b",
    r"\bwhat(?:'s| is) wrong with me\b",
    r"\bdo i have\b",
    r"\bam i (?:having|getting|suffering)\b",
    r"\bdiagnos(?:e|is|ing) me\b",
    r"\bwhat do i have\b",
    r"\bis (?:it|this) (?:cancer|a (?:heart attack|stroke|tumou?r))\b",
    r"\btell me (?:what|which) (?:disease|condition)\b",
    r"\bmy (?:symptoms?|test results?)\b.{0,40}\bmean\b",
    r"ฉันเป็น(?:โรค)?อะไร",
    r"ผมเป็น(?:โรค)?อะไร",
    r"หนูเป็น(?:โรค)?อะไร",
    r"วินิจฉัย(?:ให้|ว่า)",
    r"เป็นโรคอะไร",
)

# Requests for personalised medication dosing or prescriptions.
_MEDICATION_PATTERNS = (
    r"\bhow (?:much|many)\b.{0,40}\b(?:should|can|do) i take\b",
    r"\bwhat dose\b",
    r"\bwhat dosage\b",
    r"\bcorrect dosage\b",
    r"\bhow many (?:mg|milligrams?|tablets?|pills?|capsules?)\b",
    r"\bprescri(?:be|ption)\b.{0,30}\b(?:me|for me)\b",
    r"\bcan i take\b.{0,40}\b(?:with|together|and)\b",
    r"\bshould i (?:take|stop|increase|double)\b.{0,30}\b(?:medication|medicine|drug|dose|pill)",
    r"\bwhat (?:medicine|medication|drug|antibiotic)\s+should i\b",
    r"กินยา(?:อะไร|ตัวไหน)",
    r"ควรกินยา",
    r"กี่เม็ด",
    r"ขนาดยา",
    r"จ่ายยา(?:ให้|อะไร)",
)

# Requests for a personal treatment plan.
_TREATMENT_PATTERNS = (
    r"\b(?:treatment|therapy|care) plan for me\b",
    r"\bhow (?:do|should) i (?:treat|cure|fix) my\b",
    r"\bwhat (?:treatment|surgery) (?:do|should) i (?:need|get|have)\b",
    r"\bcure my\b",
    r"\bdiet plan for my\b",
    r"แผนการรักษา",
    r"รักษาตัวเอง(?:ยังไง|อย่างไร)",
    r"ต้องผ่าตัด(?:ไหม|มั้ย)",
)

# Diagnostic assertions in model output.
_ASSERTION_PATTERNS = (
    r"\byou (?:have|are having|have got)\s+(?!to\b|any\b|a (?:question|choice)\b)[a-z]",
    r"\byou (?:are|'re) (?:suffering from|diagnosed with|experiencing)\b",
    r"\byour (?:diagnosis|condition) is\b",
    r"\byou (?:most likely|probably|definitely) have\b",
    r"\bthis (?:means|indicates) (?:that )?you have\b",
    r"คุณ(?:กำลัง)?เป็นโรค",
    r"คุณน่าจะเป็น",
    r"คุณเป็น(?:โรค)?[ก-๙]",
)

_ASSERTION_NOTICE = (
    "> **Scope note.** The wording above came close to stating a diagnosis. "
    "This demo cannot determine what condition anyone has. Treat the answer as "
    "general information only, and ask a healthcare professional about your own "
    "situation."
)


def _compile(patterns: tuple[str, ...]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


_COMPILED = {
    "emergency": _compile(_EMERGENCY_PATTERNS),
    "crisis": _compile(_CRISIS_PATTERNS),
    "diagnosis": _compile(_DIAGNOSIS_PATTERNS),
    "medication": _compile(_MEDICATION_PATTERNS),
    "treatment": _compile(_TREATMENT_PATTERNS),
}

_COMPILED_ASSERTIONS = _compile(_ASSERTION_PATTERNS)


# --------------------------------------------------------------------------- #
# request-side check
# --------------------------------------------------------------------------- #
@dataclass
class ScopeResult:
    """Outcome of the deterministic pre-generation check.

    Attributes:
        action: ``"allow"`` runs the model unchanged, ``"reframe"`` runs the model
            with extra guidance, ``"block"`` skips the model entirely.
        categories: Every rule category that matched, for display and tests.
        notice: Markdown shown above the answer (empty when nothing matched).
        guidance: Extra instruction appended to the system prompt.
        blocked_response: Full deterministic reply used when ``action == "block"``.
    """

    action: str
    categories: list[str] = field(default_factory=list)
    notice: str = ""
    guidance: str = ""
    blocked_response: str | None = None

    @property
    def is_blocked(self) -> bool:
        return self.action == "block"


_DIAGNOSIS_GUIDANCE = (
    "The user asked you to identify what condition they have. Do not name a "
    "diagnosis for them and do not rank likelihoods for their case. Explain that "
    "the symptoms they mentioned are non-specific and can have many different "
    "causes, describe in general educational terms what those causes can include, "
    "and recommend assessment by a healthcare professional."
)

_MEDICATION_GUIDANCE = (
    "The user asked about medication dosing, drug choice, or drug interactions "
    "for themselves. Do not give a dose, a quantity, a schedule, or a "
    "recommendation to start, stop or change any medicine. You may explain in "
    "general terms what a class of medicine is used for. Direct dosing questions "
    "to a pharmacist or prescribing clinician."
)

_TREATMENT_GUIDANCE = (
    "The user asked for a treatment plan for their own situation. Do not produce "
    "one. Describe general, publicly documented management approaches for the "
    "condition as background information only, and say that the specific plan "
    "has to come from a treating clinician."
)

_DIAGNOSIS_NOTICE = (
    "> **Scope note.** This demo does not identify what condition you have. "
    "The answer below is general information, and the symptoms you described "
    "can have many different causes. Please see a healthcare professional for "
    "an assessment."
)

_MEDICATION_NOTICE = (
    "> **Scope note.** This demo does not give doses, drug choices, or "
    "interaction advice for your situation. The answer below stays at the level "
    "of general information. Ask a pharmacist or your prescriber about your own "
    "medicines."
)

_TREATMENT_NOTICE = (
    "> **Scope note.** This demo does not build treatment plans. The answer "
    "below describes general background only; your own plan has to come from a "
    "treating clinician."
)

_CATEGORY_TEXT = {
    "diagnosis": (_DIAGNOSIS_NOTICE, _DIAGNOSIS_GUIDANCE),
    "medication": (_MEDICATION_NOTICE, _MEDICATION_GUIDANCE),
    "treatment": (_TREATMENT_NOTICE, _TREATMENT_GUIDANCE),
}


def _matches(question: str, category: str) -> bool:
    return any(pattern.search(question) for pattern in _COMPILED[category])


def check_request(question: str) -> ScopeResult:
    """Classify a user question before it reaches the model.

    Args:
        question: Raw user input.

    Returns:
        A :class:`ScopeResult` describing what the app should do next.
    """
    text = (question or "").strip()

    if not text:
        return ScopeResult(
            action="block",
            categories=["empty"],
            blocked_response=(
                "Please type a health-information question first, for example "
                "*What is high blood pressure?*"
            ),
        )

    if _matches(text, "crisis"):
        return ScopeResult(
            action="block",
            categories=["crisis"],
            blocked_response=CRISIS_NOTICE,
        )

    if _matches(text, "emergency"):
        return ScopeResult(
            action="block",
            categories=["emergency"],
            blocked_response=EMERGENCY_NOTICE,
        )

    categories = [
        category
        for category in ("diagnosis", "medication", "treatment")
        if _matches(text, category)
    ]
    if not categories:
        return ScopeResult(action="allow")

    notices = [_CATEGORY_TEXT[category][0] for category in categories]
    guidance = [_CATEGORY_TEXT[category][1] for category in categories]
    return ScopeResult(
        action="reframe",
        categories=categories,
        notice="\n\n".join(notices),
        guidance=" ".join(guidance),
    )


# --------------------------------------------------------------------------- #
# response-side check
# --------------------------------------------------------------------------- #
def review_answer(answer: str) -> tuple[str, bool]:
    """Attach a correction notice when the model output reads as a diagnosis.

    Args:
        answer: Raw model output.

    Returns:
        ``(reviewed_answer, was_flagged)``.
    """
    text = answer or ""
    flagged = any(pattern.search(text) for pattern in _COMPILED_ASSERTIONS)
    if not flagged:
        return text, False
    return f"{_ASSERTION_NOTICE}\n\n{text}", True
