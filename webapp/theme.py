"""Visual theme for the Health Information Assistant demo.

A clinical-research look: calm medical blue, restrained teal accents, generous
white space, and tabular figures so benchmark numbers line up.

One deliberate constraint. The project brief says the app must not look like a
certified clinical system, so the styling stops short of a hospital-product
aesthetic: the research-demo badge and the scope disclaimer are the most
prominent things in the header, and nothing here imitates institutional
branding, accreditation marks, or a clinical record UI.
"""

from __future__ import annotations

import gradio as gr

# --------------------------------------------------------------------------- #
# palette
# --------------------------------------------------------------------------- #
# Deep, desaturated medical blue. Reads as clinical without going navy.
MEDICAL_BLUE = gr.themes.Color(
    name="medical_blue",
    c50="#eff6fb",
    c100="#d8e9f5",
    c200="#b2d2ea",
    c300="#82b4da",
    c400="#5093c6",
    c500="#2f76ae",
    c600="#215d91",
    c700="#1b4b76",
    c800="#173d60",
    c900="#15334f",
    c950="#0c2033",
)

# Muted teal for secondary surfaces - the "care" accent, used sparingly.
CLINICAL_TEAL = gr.themes.Color(
    name="clinical_teal",
    c50="#eefaf7",
    c100="#d2f2ea",
    c200="#a8e4d7",
    c300="#72cebd",
    c400="#43b09f",
    c500="#2a9285",
    c600="#1f756c",
    c700="#1c5e58",
    c800="#194c47",
    c900="#173f3c",
    c950="#0a2523",
)

MEDICAL_THEME = gr.themes.Soft(
    primary_hue=MEDICAL_BLUE,
    secondary_hue=CLINICAL_TEAL,
    neutral_hue=gr.themes.colors.slate,
    spacing_size=gr.themes.sizes.spacing_md,
    radius_size=gr.themes.sizes.radius_sm,
    text_size=gr.themes.sizes.text_md,
    # Typography is set in CUSTOM_CSS rather than through the ``font`` argument:
    # Gradio compares a custom theme against every built-in one with
    # ``Font.__eq__``, which raises when the two font lists are not both plain
    # strings. Styling the CSS variables directly avoids that comparison.
).set(
    body_background_fill="#f4f7fa",
    body_background_fill_dark="#0f1720",
    background_fill_primary="#ffffff",
    background_fill_secondary="#f8fafc",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_shadow="0 1px 2px rgba(16, 42, 67, 0.05)",
    block_label_text_weight="600",
    block_title_text_weight="600",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="#ffffff",
    button_large_radius="*radius_sm",
    button_small_radius="*radius_sm",
    input_border_color="*neutral_300",
    input_border_color_focus="*primary_500",
    panel_background_fill="*background_fill_secondary",
)


# --------------------------------------------------------------------------- #
# supplementary CSS
# --------------------------------------------------------------------------- #
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ------------------------------------------------------------ typography */
/* Falls back cleanly to the system stack when Google Fonts is unreachable. */
:root, .gradio-container {
    --font: Inter, system-ui, -apple-system, "Segoe UI", sans-serif;
    --font-mono: "JetBrains Mono", SFMono-Regular, Consolas, monospace;
}

/* ---------------------------------------------------------------- layout */
.gradio-container { max-width: 1400px !important; }

/* ---------------------------------------------------------------- header */
#app-header {
    background: linear-gradient(180deg, #ffffff 0%, #f6f9fc 100%);
    border: 1px solid var(--neutral-200);
    border-top: 3px solid var(--primary-600);
    border-radius: var(--radius-sm);
    padding: 18px 22px 16px;
    margin-bottom: 14px;
}
#app-header .eyebrow {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--primary-700);
    margin: 0 0 6px;
}
#app-header h1 {
    font-size: 25px;
    font-weight: 650;
    line-height: 1.25;
    letter-spacing: -0.015em;
    color: var(--neutral-900);
    margin: 0 0 4px;
}
#app-header .subtitle {
    font-size: 13.5px;
    color: var(--neutral-600);
    margin: 0;
}
#app-header .badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin: 12px 0 0;
}
#app-header .badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.03em;
    padding: 3px 9px;
    border-radius: 999px;
    border: 1px solid transparent;
}
#app-header .badge.research {
    background: #fdf4e3;
    border-color: #eccd8f;
    color: #7a5410;
}
#app-header .badge.neutral {
    background: var(--neutral-100);
    border-color: var(--neutral-300);
    color: var(--neutral-700);
}

/* -------------------------------------------------------------- notices */
#safety-banner {
    display: flex;
    gap: 11px;
    align-items: flex-start;
    background: #fbf6ec;
    border: 1px solid #e8d4a8;
    border-left: 4px solid #c8912f;
    border-radius: var(--radius-sm);
    padding: 11px 15px;
    margin-bottom: 16px;
    font-size: 13.5px;
    line-height: 1.55;
    color: #5c451c;
}
#safety-banner .icon { font-size: 15px; line-height: 1.4; }
#safety-banner strong { color: #4a3714; }

/* ------------------------------------------------------------- side rail */
#control-rail {
    background: #ffffff;
    border: 1px solid var(--neutral-200);
    border-radius: var(--radius-sm);
    padding: 14px 14px 6px;
}
#control-rail .block { border: none; box-shadow: none; }

/* Live metrics read as a clinical readout panel. */
#metrics-panel {
    background: var(--primary-50);
    border: 1px solid var(--primary-200);
    border-radius: var(--radius-sm);
    padding: 12px 14px;
    margin-top: 4px;
}
#metrics-panel h3 {
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--primary-800) !important;
    margin: 0 0 6px !important;
}
#metrics-panel ul { margin: 4px 0 0; padding-left: 16px; }
#metrics-panel li { font-size: 12.5px; line-height: 1.7; }
#metrics-panel strong { color: var(--neutral-800); }
#metrics-panel code {
    font-size: 11px;
    font-variant-numeric: tabular-nums;
}

/* ------------------------------------------------------------------ chat */
#chat-window { border-radius: var(--radius-sm); }

/* Deterministic scope notes must stand out from model prose. */
#chat-window blockquote {
    border-left: 3px solid var(--secondary-500);
    background: var(--secondary-50);
    margin: 0 0 10px;
    padding: 9px 13px;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    color: var(--neutral-700);
    font-size: 13.5px;
}
#chat-window blockquote strong { color: var(--secondary-800); }
#chat-window h3 { font-size: 15px; margin-top: 2px; }

/* --------------------------------------------------------- benchmark tab */
#benchmark-tab table {
    font-variant-numeric: tabular-nums;
    font-size: 13px;
}
#benchmark-tab h2 {
    font-size: 17px;
    font-weight: 650;
    color: var(--neutral-900);
    border-bottom: 2px solid var(--primary-100);
    padding-bottom: 5px;
    margin: 26px 0 4px;
}
#benchmark-tab h3 { font-size: 14.5px; font-weight: 650; margin-top: 18px; }
#benchmark-tab h4 { font-size: 13px; font-weight: 650; }
#benchmark-tab blockquote {
    border-left: 3px solid var(--primary-400);
    background: var(--primary-50);
    padding: 9px 13px;
    margin: 10px 0;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    font-size: 13px;
}

/* Result maturity is a caveat, not a headline: amber, not brand blue. */
#maturity-banner {
    background: #fbf6ec;
    border: 1px solid #e8d4a8;
    border-radius: var(--radius-sm);
    padding: 13px 16px;
    margin-bottom: 6px;
}
#maturity-banner h3 {
    font-size: 14px !important;
    font-weight: 700 !important;
    color: #7a5410 !important;
    margin: 0 0 7px !important;
}
#maturity-banner ul { margin: 6px 0 0; }
#maturity-banner li { font-size: 12.5px; line-height: 1.65; }

/* Environment tables read as a spec readout, not a data grid: no inner rules,
   hairline row separators, label column muted. */
#benchmark-env table, #demo-runtime table {
    width: auto;
    min-width: 430px;
    border-collapse: collapse;
    border: 1px solid var(--neutral-200);
    border-radius: var(--radius-sm);
    overflow: hidden;
}
#benchmark-env th, #demo-runtime th {
    text-align: left;
    padding: 7px 18px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--neutral-600);
    background: var(--neutral-50);
    border: none;
    border-bottom: 1px solid var(--neutral-200);
}
#benchmark-env td, #demo-runtime td {
    padding: 6px 18px;
    font-size: 13px;
    border: none;
    border-bottom: 1px solid var(--neutral-100);
}
#benchmark-env tr:last-child td, #demo-runtime tr:last-child td {
    border-bottom: none;
}
#benchmark-env td:first-child, #demo-runtime td:first-child {
    color: var(--neutral-600);
    white-space: nowrap;
    width: 190px;
}
#benchmark-env td:last-child, #demo-runtime td:last-child {
    font-family: var(--font-mono);
    font-size: 12.5px;
    color: var(--neutral-800);
}

#problem-banner {
    background: #fdf1f0;
    border: 1px solid #e9b8b3;
    border-left: 4px solid #c0453a;
    border-radius: var(--radius-sm);
    padding: 12px 15px;
    margin-bottom: 12px;
}
#problem-banner h3 { color: #8c2f26 !important; font-size: 14px !important; }

/* ------------------------------------------------------------------ misc */
footer { display: none !important; }
"""


HEADER_HTML = """
<div id="app-header">
  <p class="eyebrow">MedPubCodex &middot; Quantization Benchmark</p>
  <h1>Quantized Medical LLM &mdash; Health Information Assistant</h1>
  <p class="subtitle">
    Quantized Gemma&nbsp;4 and MedGemma checkpoints, served behind a
    health-information interface.
  </p>
  <div class="badge-row">
    <span class="badge research">Research Demo</span>
    <span class="badge neutral">Not a diagnostic tool</span>
    <span class="badge neutral">BitsAndBytes INT8 / INT4-NF4</span>
    <span class="badge neutral">Single-turn inference</span>
  </div>
</div>
"""

SAFETY_BANNER_HTML = """
<div id="safety-banner">
  <span class="icon">&#9888;</span>
  <span>
    <strong>For research and educational purposes only.</strong>
    This application does not provide medical diagnosis and does not replace a
    qualified healthcare professional. Answers are generated by a quantized
    language model and may be incorrect.
  </span>
</div>
"""
