"""BIRD Text-to-SQL evaluation harness for the DataAgent project.

Supports the three evaluation settings from Research_Plan.md section six:
no knowledge (lower bound), oracle evidence (upper bound), and flat-RAG
retrieval (the KAT-SQL-style baseline; currently a stub).

P1a (2026-05-20): adds `extraction` (LLM-driven BIRD-evidence -> v2-typed-node
extractor) and `extraction_eval` (per-type F1 / grounding accuracy scoring).

P1b (2026-05-21): adds `joint_graph` (per-DB JointGraph: v2 typed nodes +
schema columns + FK edges + indexes), built from a v2 annotation file.
"""
