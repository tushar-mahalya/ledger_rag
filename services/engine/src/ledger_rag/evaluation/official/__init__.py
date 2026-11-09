"""Official benchmark metrics, so reported scores are comparable with published results.

``finqa`` wraps the vendored upstream FinQA executor (``_finqa_upstream``) for
execution accuracy and adds the Program-of-Thoughts tolerant comparison that LLM
papers report. Never edit vendored files; wrap them instead.
"""
