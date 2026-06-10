"""
Stryker AI Diagnostic Assistant — Diagnostic Inference Engine
Provides LLM-assisted diagnostic recommendations from imaging and sensor data.
"""
import os
import json
from typing import Optional


# NOTE: CCI-AI-001 — LLM client initialized without governance manifest.
# Model: claude-sonnet-4-6 (unregistered, no approval record, no traceability ID)
# Human owner: not declared. Approval status: not submitted.
import anthropic

_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
_MODEL = "claude-sonnet-4-6"
_SYSTEM_PROMPT = """You are a clinical diagnostic assistant supporting Stryker surgical teams.
You analyze imaging metadata and sensor telemetry to surface anomaly indicators.
You do not make final clinical decisions — outputs are advisory only."""


def analyze_imaging_metadata(metadata: dict, patient_context: Optional[dict] = None) -> dict:
    """
    Run LLM inference over imaging metadata to flag anomalies.
    Returns structured finding with confidence score and recommendation.
    """
    prompt = f"""Imaging metadata received from surgical suite:
{json.dumps(metadata, indent=2)}

Patient context (de-identified):
{json.dumps(patient_context or {}, indent=2)}

Identify any anomalies or patterns that warrant clinical review.
Return JSON with: anomalies (list), confidence (0-1), recommendation (string), urgency (low/medium/high/critical).
"""
    response = _client.messages.create(
        model=_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        return json.loads(response.content[0].text)
    except (json.JSONDecodeError, IndexError):
        return {"raw": response.content[0].text, "parse_error": True}


def summarize_sensor_trend(readings: list[dict], window_hours: int = 24) -> str:
    """
    Summarize a time-series of device sensor readings using the LLM.
    Used by clinical dashboard for daily review.
    """
    prompt = f"""Summarize the following {window_hours}-hour sensor trend for clinical review.
Readings (most recent first):
{json.dumps(readings[:50], indent=2)}

Produce a 2-3 sentence clinical summary suitable for a charge nurse briefing.
Flag any values outside normal operating range."""

    response = _client.messages.create(
        model=_MODEL,
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
