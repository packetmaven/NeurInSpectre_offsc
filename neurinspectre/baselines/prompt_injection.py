"""
Prompt-injection baseline scanners.

Baselines requested for Issue 4 (Attention §5.7):
- LLM Guard (Protect AI) prompt injection scanner
- Rebuff (Protect AI) prompt injection detector (requires API keys)
- Spotlighting (Hines et al. 2024) input transformation (no external deps)

These are intentionally implemented as *adapters*:
- LLM Guard and Rebuff are external projects; we call their public APIs when present.
- Spotlighting is implemented locally as a deterministic input transform.
"""

from __future__ import annotations

import base64
import codecs
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PromptScanResult:
    baseline: str
    ok: bool
    risk_score: Optional[float] = None
    sanitized: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline,
            "ok": bool(self.ok),
            "risk_score": None if self.risk_score is None else float(self.risk_score),
            "sanitized": self.sanitized,
            "details": dict(self.details or {}),
        }


def scan_llm_guard(
    prompt: str,
    *,
    threshold: float = 0.5,
    match_type: str = "full",
) -> PromptScanResult:
    """
    Run LLM Guard's PromptInjection scanner (if installed).

    match_type: "full" (default) or "sentence" (LLM Guard MatchType)
    """
    try:
        from llm_guard.input_scanners import PromptInjection
        from llm_guard.input_scanners.prompt_injection import MatchType
    except Exception as exc:
        raise ImportError(
            "LLM Guard baseline requires the optional dependency `llm-guard`.\n"
            "Install: pip install llm-guard"
        ) from exc

    mt = str(match_type or "full").strip().lower()
    if mt in {"full"}:
        mt_enum = MatchType.FULL
    elif mt in {"sentence", "sentences"}:
        mt_enum = MatchType.SENTENCE
    else:
        raise ValueError(f"Unknown match_type={match_type!r}. Expected: 'full' or 'sentence'.")

    scanner = PromptInjection(threshold=float(threshold), match_type=mt_enum)
    sanitized_prompt, is_valid, risk_score = scanner.scan(str(prompt))
    # LLM Guard returns `is_valid=True` when the prompt is acceptable (not injected).
    ok = bool(is_valid)
    return PromptScanResult(
        baseline="llm_guard",
        ok=ok,
        risk_score=float(risk_score) if risk_score is not None else None,
        sanitized=str(sanitized_prompt),
        details={"threshold": float(threshold), "match_type": str(mt)},
    )


def scan_rebuff(
    prompt: str,
    *,
    openai_api_key: str,
    pinecone_api_key: str,
    pinecone_index: str,
    openai_model: Optional[str] = None,
) -> PromptScanResult:
    """
    Run Rebuff's detector (if installed and keys provided).

    Rebuff is networked by design; we keep it explicitly opt-in.
    """
    try:
        from rebuff import RebuffSdk
    except Exception as exc:
        raise ImportError(
            "Rebuff baseline requires the optional dependency `rebuff`.\n"
            "Install: pip install rebuff"
        ) from exc

    rb = RebuffSdk(
        openai_api_key,
        pinecone_api_key,
        pinecone_index,
        openai_model or "gpt-3.5-turbo",
    )
    result = rb.detect_injection(str(prompt))

    injection = bool(getattr(result, "injection_detected", False))
    # Rebuff has multiple layers; many results expose a numeric score and vector matches.
    score = getattr(result, "injection_score", None)
    try:
        score_f = float(score) if score is not None else None
    except Exception:
        score_f = None

    return PromptScanResult(
        baseline="rebuff",
        ok=not injection,
        risk_score=score_f,
        sanitized=str(prompt),
        details={
            "injection_detected": injection,
            "raw": {k: getattr(result, k) for k in dir(result) if k.endswith("score") and not k.startswith("_")},
        },
    )


def spotlight_encode_untrusted(text: str, *, encoding: str = "base64") -> str:
    """
    Spotlighting transform: encode untrusted content to create a strong boundary
    signal for an LLM (per Hines et al., 2024).

    This function does not call any model; it only produces the transformed text.
    """
    enc = str(encoding or "base64").strip().lower()
    raw = str(text or "")
    if enc == "base64":
        b = raw.encode("utf-8", errors="ignore")
        return base64.b64encode(b).decode("ascii")
    if enc in {"rot13", "rot-13"}:
        return codecs.encode(raw, "rot_13")
    raise ValueError(f"Unknown encoding={encoding!r}. Expected 'base64' or 'rot13'.")


def spotlight_wrap_prompt(
    *,
    system_instructions: str,
    untrusted_text: str,
    encoding: str = "base64",
) -> str:
    """
    Construct a "spotlighted" prompt with an explicit system-level boundary.
    """
    encoded = spotlight_encode_untrusted(untrusted_text, encoding=encoding)
    return (
        f"{system_instructions.strip()}\n\n"
        "UNTRUSTED_CONTENT_ENCODING=" + str(encoding).strip().lower() + "\n"
        "UNTRUSTED_CONTENT_BEGIN\n"
        f"{encoded}\n"
        "UNTRUSTED_CONTENT_END\n"
    )

