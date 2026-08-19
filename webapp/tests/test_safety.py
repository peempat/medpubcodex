"""Tests for the deterministic scope layer.

These check that the rule layer routes questions the way the demo needs. They do
not establish that the demo is medically safe.
"""

import pytest

import safety


# --------------------------------------------------------------------------- #
# general information is allowed through untouched
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "question",
    [
        "What does hypertension mean?",
        "What is high blood pressure?",
        "What is diabetes mellitus?",
        "What foods are high in fiber?",
        "How does the immune system work?",
        "What is the difference between type 1 and type 2 diabetes?",
        "ความดันโลหิตสูงคืออะไร",
    ],
)
def test_general_information_is_allowed(question):
    result = safety.check_request(question)
    assert result.action == "allow"
    assert result.categories == []
    assert result.notice == ""
    assert result.guidance == ""


# --------------------------------------------------------------------------- #
# diagnosis requests are redirected, not answered as a diagnosis
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "question",
    [
        "What disease do I have?",
        "I have headache and dizziness. What disease do I have?",
        "Do I have diabetes?",
        "What's wrong with me?",
        "Can you diagnose me based on these symptoms?",
        "ฉันเป็นโรคอะไร",
    ],
)
def test_diagnosis_requests_are_reframed(question):
    result = safety.check_request(question)
    assert result.action == "reframe"
    assert "diagnosis" in result.categories
    # The user still gets an answer, but scoped and labelled.
    assert not result.is_blocked
    assert result.notice
    assert "do not name a diagnosis" in result.guidance.lower()
    assert "healthcare professional" in result.guidance.lower()


def test_diagnosis_guidance_reaches_the_system_prompt():
    """The guidance has to be injected, not merely returned."""
    import prompts

    result = safety.check_request("I have headache and dizziness. What disease do I have?")
    use_case = prompts.USE_CASES_BY_KEY["general_qa"]
    system_prompt = prompts.build_system_prompt(use_case, result.guidance)
    assert "Do not name a diagnosis" in system_prompt


# --------------------------------------------------------------------------- #
# medication dosing is redirected
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "question",
    [
        "How much paracetamol should I take?",
        "What dose of metformin do I need?",
        "How many mg of ibuprofen can I take for back pain?",
        "Can I take amoxicillin with my blood pressure medicine?",
        "Should I stop my medication if I feel better?",
        "ควรกินยาลดความดันกี่เม็ด",
    ],
)
def test_dosage_requests_are_reframed(question):
    result = safety.check_request(question)
    assert result.action == "reframe"
    assert "medication" in result.categories
    guidance = result.guidance.lower()
    assert "do not give a dose" in guidance
    assert "pharmacist" in guidance


def test_dosage_notice_states_the_limit_to_the_user():
    result = safety.check_request("How many mg of ibuprofen should I take?")
    assert "does not give doses" in result.notice


# --------------------------------------------------------------------------- #
# treatment plans are redirected
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "question",
    [
        "How should I treat my high blood pressure?",
        "What treatment do I need for my back pain?",
        "Build a diet plan for my diabetes.",
    ],
)
def test_treatment_plan_requests_are_reframed(question):
    result = safety.check_request(question)
    assert result.action == "reframe"
    assert "treatment" in result.categories


# --------------------------------------------------------------------------- #
# urgent symptoms bypass the model entirely
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "question",
    [
        "I have crushing chest pain and my left arm is numb.",
        "My father is unconscious and not breathing.",
        "I am coughing up blood, what should I do?",
        "เจ็บหน้าอกมาก หายใจไม่ออก",
    ],
)
def test_urgent_symptoms_are_blocked_with_care_advice(question):
    result = safety.check_request(question)
    assert result.is_blocked
    assert result.categories == ["emergency"]
    body = result.blocked_response
    assert "1669" in body or "emergency number" in body.lower()
    # It must not attempt a cause.
    assert "you have" not in body.lower()


def test_self_harm_is_routed_to_support_not_to_the_model():
    result = safety.check_request("I want to hurt myself")
    assert result.is_blocked
    assert result.categories == ["crisis"]
    assert "1323" in result.blocked_response


# --------------------------------------------------------------------------- #
# empty input
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("question", ["", "   ", "\n\t ", None])
def test_empty_input_is_blocked(question):
    result = safety.check_request(question)
    assert result.is_blocked
    assert result.categories == ["empty"]
    assert "question" in result.blocked_response.lower()


# --------------------------------------------------------------------------- #
# response-side review
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "answer",
    [
        "Based on your symptoms, you have hypertension.",
        "You are suffering from type 2 diabetes.",
        "Your diagnosis is migraine.",
        "This means that you have an infection.",
        "คุณเป็นโรคความดันโลหิตสูง",
    ],
)
def test_diagnostic_assertions_in_output_are_flagged(answer):
    reviewed, flagged = safety.review_answer(answer)
    assert flagged
    assert "close to stating a diagnosis" in reviewed
    assert answer in reviewed  # the original text is kept, not silently dropped


@pytest.mark.parametrize(
    "answer",
    [
        "Hypertension means blood pressure is persistently higher than normal.",
        "Foods high in fiber include beans, oats, and most vegetables.",
        "You have to consider that many conditions share these symptoms.",
        "If you have any questions, ask a healthcare professional.",
    ],
)
def test_general_answers_are_not_flagged(answer):
    reviewed, flagged = safety.review_answer(answer)
    assert not flagged
    assert reviewed == answer


def test_review_handles_empty_output():
    reviewed, flagged = safety.review_answer("")
    assert reviewed == ""
    assert not flagged
