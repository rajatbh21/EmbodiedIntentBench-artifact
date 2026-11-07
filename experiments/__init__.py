"""Experimental tooling that lives *outside* the Mission Routing Benchmark runtime core.

Nothing under ``experiments/`` is imported by the benchmark package. It is where real
datasets and real models are wired up, so the core keeps its zero-dependency runtime and its
model-agnostic boundary. Code here may declare heavy optional dependencies of its own.
"""
