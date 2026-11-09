"""High-precision out-of-scope patterns shared by routing and history screening."""

import re

ADVICE = re.compile(
    r"\bshould i (?:buy|sell|invest|hold|short)\b|\bprice target\b|\bstock (?:tip|pick)s?\b"
    r"|\b(?:predict|forecast)\w* (?:the |next year'?s? |future )?(?:stock|share) price\b"
    r"|\bwill (?:the )?(?:stock|shares?) (?:go up|go down|rise|fall)\b|\bbest stock\b",
    re.I,
)
OFF_TOPIC = re.compile(
    r"\b(?:weather|recipe|poem|song|lyrics|joke|horoscope|movie|sports? score)\b"
    r"|\bwrite (?:me )?(?:a|an) (?:story|essay|poem)\b",
    re.I,
)
INJECTION = re.compile(
    r"\bignore (?:all |any )?(?:previous|prior|above) instructions\b|\bsystem prompt\b"
    r"|\bdeveloper mode\b|\bjailbreak\b",
    re.I,
)


def flagged(text: str) -> bool:
    """True for prompt-injection or personal investment-advice text."""
    return bool(INJECTION.search(text) or ADVICE.search(text))
