"""Compatibility shim for the legacy ``pipeline.*`` import path.

After the core refactor the real step modules live in ``core.steps`` (and
``core.utils``). The SQS worker still imports them as ``pipeline.<name>`` (e.g.
``from pipeline.tts import synthesize``), so here we alias each core module
into ``sys.modules`` under the ``pipeline.<name>`` name. That makes the dotted
submodule imports resolve to the exact same module objects — no code change in
``worker.py`` required.

Optional-dependency modules (e.g. ``replicate_fallback`` needs the ``replicate``
package, only used on CPU/fallback hosts) are imported lazily: if the dep is
missing we skip the alias instead of breaking the whole shim. ``worker.py`` only
imports those under the matching runtime branch, so this is safe.
"""

import importlib
import sys

_STEP_MODULES = [
    "animate",
    "avatar_gen",
    "broll",
    "codeformer_polish",
    "compositor",
    "echomimic",
    "face_composite",
    "lipsync",
    "mimicmotion",
    "pose_extract",
    "pose_extract_echomimic",
    "replicate_fallback",
    "sonic_lipsync",
    "soul_id",
    "text_to_video",
    "tts",
    "video_postproc",
    "video_sr",
    "voiceclone",
]

for _name in _STEP_MODULES:
    try:
        _mod = importlib.import_module(f"core.steps.{_name}")
    except Exception:
        # Optional dependency not installed on this host; skip the alias.
        continue
    sys.modules[f"{__name__}.{_name}"] = _mod
    globals()[_name] = _mod

try:
    _gpu_env = importlib.import_module("core.utils.gpu_env")
    sys.modules[f"{__name__}.gpu_env"] = _gpu_env
    globals()["gpu_env"] = _gpu_env
except Exception:
    pass
