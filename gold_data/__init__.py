"""Gold-data tooling for the recursive-improvement loop.

Dev-time only: downloads public PII datasets, maps them to GIRP elements, and writes a
versioned local gold set. Named `gold_data` (not `datasets`) so it never shadows the
HuggingFace `datasets` import. Production code (girp.py / aupii.py) never imports this
package, so the offline runtime guarantee is unaffected.
"""
