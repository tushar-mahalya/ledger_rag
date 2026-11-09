"""Program-of-thought financial reasoning: the model plans, deterministic code computes.

A reader model never does arithmetic. It writes a short FinQA-style program over
numbers it read from the evidence; the program is parsed and bounded
(:mod:`.dsl`), every literal is checked against the evidence (:mod:`.grounding`),
executed deterministically, and independent samples vote on the executed result
(:mod:`.consistency`). Similar solved training problems are retrieved as
in-context exemplars (:mod:`.exemplars`). This is what lets a small reader compete:
arithmetic and transcription errors are removed by construction, and sampling buys
accuracy with parallel, prefix-cached decoding instead of a larger model.
"""
