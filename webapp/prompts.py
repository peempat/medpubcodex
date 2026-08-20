"""System prompts for the Health Information Assistant demo.

Prompting alone is not treated as a safety mechanism here. The deterministic
rules in :mod:`safety` run before and after generation; these prompts set the
expected behaviour for the ordinary case.

Generation is single-turn by design: the current question plus the system prompt
for the selected use case. Chat history stays visible in the UI but is not fed
back into the model, because multi-turn behaviour was not part of the benchmark
and has not been evaluated here.
"""

from __future__ import annotations

from dataclasses import dataclass

BASE_SYSTEM_PROMPT = """You are a health-information assistant in a research demo.
You provide general, educational health information only.

How to answer:
- Answer in the same language the user wrote in.
- Explain concepts in plain language and define medical terms when you use them.
- Keep answers focused and structured; use short paragraphs or bullet points.
- Say plainly when something is uncertain, varies between people, or is disputed.
- Prefer widely accepted, general information over unusual or contested claims.

What you must not do:
- Do not tell the user what condition they have, and do not rank likely diagnoses
  for their case. Symptoms have many possible causes.
- Do not give medication doses, quantities, schedules, or drug choices for an
  individual, and do not advise starting, stopping, or changing any medicine.
- Do not write a personal treatment plan or a personalised therapeutic diet.
- Do not claim to be a doctor or to have examined anyone.

When a question is about the user's own health situation, give the general
information behind it and recommend assessment by a healthcare professional.
If the question suggests symptoms that could need urgent care, say so and
recommend appropriate professional or emergency care rather than trying to work
out the cause."""


@dataclass(frozen=True)
class UseCase:
    """One selectable assistant mode."""

    key: str
    label: str
    description: str
    instruction: str
    example: str


USE_CASES: tuple[UseCase, ...] = (
    UseCase(
        key="general_qa",
        label="General Health Q&A",
        description="Broad, educational answers to everyday health questions.",
        example="What is high blood pressure?",
        instruction=(
            "Answer this general health question for a non-specialist reader. "
            "Give the core explanation first, then the context that makes it "
            "useful (why it matters, what is generally understood about it). "
            "Keep it educational and general - this is not about any one person."
        ),
    ),
    UseCase(
        key="explain_term",
        label="Explain Medical Term",
        description="Plain-language definition of a medical word or phrase.",
        example="What does hypertension mean?",
        instruction=(
            "Explain the medical term the user asked about. Cover: a one-sentence "
            "plain-language definition, where the word comes from or how it is "
            "used in practice, and any everyday word that means the same thing. "
            "Define the term itself - do not assess whether the user has it."
        ),
    ),
    UseCase(
        key="disease_info",
        label="Disease Information",
        description="General overview of a named condition.",
        example="What is diabetes mellitus?",
        instruction=(
            "Give a general overview of the condition the user named, using these "
            "sections:\n"
            "1. What the condition is\n"
            "2. Common symptoms reported in general\n"
            "3. Common risk factors\n"
            "4. General health information (how it is usually monitored or "
            "managed at a general level, and when people are advised to seek "
            "medical care)\n\n"
            "Describe the condition in general terms. Do not assess, suggest, or "
            "imply that the user has it, even if they describe symptoms."
        ),
    ),
    UseCase(
        key="nutrition",
        label="Nutrition Information",
        description="General educational nutrition information.",
        example="What foods are high in fiber?",
        instruction=(
            "Give general educational nutrition information. Name typical foods "
            "or nutrient groups and explain the general role they play. You may "
            "mention widely published general guidance. Do not build a "
            "personalised diet, a meal plan, or a therapeutic diet for a medical "
            "condition, and do not give individual calorie or nutrient targets. "
            "Point to a registered dietitian or clinician for anything personal."
        ),
    ),
)

USE_CASES_BY_KEY = {use_case.key: use_case for use_case in USE_CASES}
USE_CASES_BY_LABEL = {use_case.label: use_case for use_case in USE_CASES}
DEFAULT_USE_CASE_KEY = "general_qa"


def get_use_case(key_or_label: str) -> UseCase:
    """Resolve a dropdown label or key to a :class:`UseCase`.

    Raises:
        KeyError: if the selection is not a known use case.
    """
    if key_or_label in USE_CASES_BY_KEY:
        return USE_CASES_BY_KEY[key_or_label]
    if key_or_label in USE_CASES_BY_LABEL:
        return USE_CASES_BY_LABEL[key_or_label]
    raise KeyError(f"Unknown use case: {key_or_label!r}")


def use_case_labels() -> list[str]:
    return [use_case.label for use_case in USE_CASES]


def build_system_prompt(use_case: UseCase, scope_guidance: str = "") -> str:
    """Compose the system prompt for one turn.

    Args:
        use_case: Selected assistant mode.
        scope_guidance: Extra instruction from :func:`safety.check_request`.
    """
    parts = [BASE_SYSTEM_PROMPT, f"Task for this answer:\n{use_case.instruction}"]
    if scope_guidance:
        parts.append(f"Additional constraint for this specific question:\n{scope_guidance}")
    return "\n\n".join(parts)


def build_messages(question: str, use_case: UseCase, scope_guidance: str = "") -> list[dict]:
    """Build the single-turn chat messages passed to the processor.

    History is intentionally not included; see the module docstring.
    """
    return [
        {"role": "system", "content": build_system_prompt(use_case, scope_guidance)},
        {"role": "user", "content": question.strip()},
    ]
