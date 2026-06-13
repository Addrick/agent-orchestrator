# src/personas/__init__.py
"""Persona persistence package (DP-203).

`src.personas.store` owns the persona save file (load/save user personas,
system-persona loading, and the model-catalog cache that shares the same
JSON file). The Persona domain object itself still lives in `src.persona`;
moving it here is optional future work (see extensibility sprints plan).
"""
