"""
Report Generation — turns structured pipeline output into a readable report
using the Gemini API.

Uses Gemini's built-in "Grounding with Google Search" tool so the model can,
when useful, search for current cost/depreciation reference points instead
of relying only on the local price table (see cost_estimation/estimator.py
and repair_advisor/advisor.py).
"""

import os


REPORT_SYSTEM_PROMPT = """\
You are an assistant that writes clear, professional vehicle damage
assessment reports for end users (car owners, insurance adjusters, or
resale buyers/sellers).

Rules you must follow:
- Clearly separate MEASURED facts (from image analysis: damaged part,
  damage type, % area damaged) from ESTIMATED figures (hidden damage risk,
  cost estimates, resale depreciation).
- For every estimated figure, state its confidence level and basis in
  plain language.
- Never claim certainty about anything not visible in the photo.
- End every report with a one-line reminder that this is a preliminary
  automated assessment and not a substitute for an in-person inspection by
  a certified appraiser.
- Keep the tone factual and neutral, not alarmist.
"""


def build_client():
    """
    Builds a Gemini client with Google Search grounding enabled.
    Requires the GEMINI_API_KEY environment variable to be set.
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")

    client = genai.Client(api_key=api_key)
    search_tool = types.Tool(google_search=types.GoogleSearch())
    return client, search_tool


def generate_report(pipeline_output: dict, model: str = "gemini-2.5-flash") -> str:
    """
    Generates a natural-language report from the pipeline's structured
    output (fusion results, hidden damage risk, cost estimates, repair vs.
    replace recommendation).
    """
    from google.genai import types

    client, search_tool = build_client()

    prompt = (
        f"{REPORT_SYSTEM_PROMPT}\n\n"
        f"Here is the structured assessment data:\n{pipeline_output}\n\n"
        "Write the final report for the end user."
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(tools=[search_tool]),
    )
    return response.text
