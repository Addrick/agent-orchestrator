# src/engine/providers/local_backend.py
"""Which server the local provider talks to, switchable at runtime (DP-438).

Resolution order, first hit wins:

  1. temporary — set from the portal's backend switch; in memory only, so a
     restart drops it.
  2. saved     — the portal's "Save"; a small JSON file at
     ``LOCAL_BACKEND_FILE``, read lazily on first use.
  3. env / default — ``LOCAL_LLM_URL`` and ``KOBOLD_PROGRESS_URL``, as before.

A backend is a model URL plus the optional kcpp-progress sidecar URL for that
same box, because the sidecar reports one specific KoboldCPP's stdout — keeping
the global sidecar after a swap would draw the old box's ingest bar under the
new one. A target with no sidecar falls through to Strata ``/metrics``.

Switching needs no reconnect logic here: `local.get_local_client` already
rebuilds its client whenever `local_base_url()` changes (DP-435).
"""

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from config import global_config
from src.utils.atomic_json import write_json_atomic

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalBackend:
    url: str                            # normalized, no trailing /v1
    progress_url: Optional[str] = None  # kcpp-progress sidecar for this box
    label: Optional[str] = None


_temporary: Optional[LocalBackend] = None
_saved: Optional[LocalBackend] = None
_saved_loaded = False


def normalize_local_url(raw: str) -> str:
    """Base URL without a trailing ``/v1``. Lenient: repairs a doubled scheme
    from copy-paste and adds ``http://`` when none is given."""
    raw = (raw or "").strip().rstrip("/")
    m = re.match(r"^(https?://)+", raw)
    if m:
        scheme = "https://" if "https://" in m.group(0) else "http://"
        raw = scheme + raw[m.end():]
    elif not raw.startswith(("http://", "https://")):
        raw = f"http://{raw}"
    raw = raw.rstrip("/")
    if raw.endswith("/v1"):
        raw = raw[:-3].rstrip("/")
    return raw


def validate_local_url(raw: str) -> str:
    """`normalize_local_url`, then reject what cannot be a server address.
    Raises ValueError with a message safe to show the operator."""
    url = normalize_local_url(raw)
    parsed = urlparse(url)
    # A bare scheme ("http://") would normalize to the host "http".
    bare = re.sub(r"^(https?:/*)+", "", (raw or "").strip()).strip("/")
    if not bare or not parsed.hostname or any(c.isspace() for c in url):
        raise ValueError("not a valid http(s) URL")
    try:
        parsed.port
    except ValueError:
        raise ValueError("invalid port") from None
    return url


def presets() -> List[LocalBackend]:
    """`LOCAL_LLM_PRESETS`, parsed. Bad entries are logged and skipped — a typo
    in one preset must not take the switch (or startup) down."""
    try:
        raw = json.loads(global_config.LOCAL_LLM_PRESETS or "[]")
    except json.JSONDecodeError:
        logger.warning("LOCAL_LLM_PRESETS is not valid JSON; no presets offered")
        return []
    out: List[LocalBackend] = []
    for entry in raw if isinstance(raw, list) else []:
        try:
            out.append(LocalBackend(
                url=validate_local_url(entry["url"]),
                progress_url=validate_local_url(entry["progress_url"]) if entry.get("progress_url") else None,
                label=str(entry.get("label") or "") or None,
            ))
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"LOCAL_LLM_PRESETS: skipping entry {entry!r}: {e}")
    return out


def _load_saved() -> Optional[LocalBackend]:
    global _saved, _saved_loaded
    if _saved_loaded:
        return _saved
    _saved_loaded = True
    path = global_config.LOCAL_BACKEND_FILE
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        _saved = LocalBackend(
            url=validate_local_url(data["url"]),
            progress_url=validate_local_url(data["progress_url"]) if data.get("progress_url") else None,
            label=data.get("label") or None,
        )
        logger.info(f"Local backend: using saved {_saved.url} ({path})")
    except Exception as e:
        # A corrupt file falls back to env rather than failing every local turn.
        logger.warning(f"Local backend: ignoring unreadable {path}: {e}")
        _saved = None
    return _saved


def _env_backend() -> Tuple[LocalBackend, str]:
    env_url = os.environ.get("LOCAL_LLM_URL")
    source = "env" if env_url else "default"
    url = normalize_local_url(env_url or global_config.LOCAL_LLM_URL)
    progress = os.environ.get("KOBOLD_PROGRESS_URL", global_config.KOBOLD_PROGRESS_URL) or ""
    return LocalBackend(url=url, progress_url=progress.rstrip("/") or None), source


def active() -> Tuple[LocalBackend, str]:
    """The backend in effect and where it came from:
    ``temporary`` | ``saved`` | ``env`` | ``default``."""
    if _temporary is not None:
        return _temporary, "temporary"
    saved = _load_saved()
    if saved is not None:
        return saved, "saved"
    return _env_backend()


def apply(url: str, progress_url: Optional[str] = None, label: Optional[str] = None) -> LocalBackend:
    """Switch to ``url`` until restart (or Save). A URL matching a preset
    inherits that preset's label and sidecar unless given explicitly."""
    global _temporary
    norm = validate_local_url(url)
    match = next((p for p in presets() if p.url == norm), None)
    backend = LocalBackend(
        url=norm,
        progress_url=validate_local_url(progress_url) if progress_url else (match.progress_url if match else None),
        label=label or (match.label if match else None),
    )
    old, _ = active()
    _temporary = backend
    logger.info(f"Local backend switched: {old.url} -> {backend.url} (temporary)")
    return backend


def save() -> None:
    """Persist the active temporary backend. No-op when nothing is temporary."""
    global _temporary, _saved, _saved_loaded
    if _temporary is None:
        return
    path = global_config.LOCAL_BACKEND_FILE
    write_json_atomic(path, asdict(_temporary))
    _saved, _saved_loaded, _temporary = _temporary, True, None
    logger.info(f"Local backend saved: {_saved.url} ({path})")


def reset() -> None:
    """Drop the temporary and the saved backend; env/default applies again."""
    global _temporary, _saved, _saved_loaded
    _temporary, _saved, _saved_loaded = None, None, True
    path = global_config.LOCAL_BACKEND_FILE
    if os.path.exists(path):
        os.remove(path)
    logger.info("Local backend reset to LOCAL_LLM_URL")


def state() -> Dict[str, Any]:
    """The portal's view. Sidecar URLs stay server-side — the browser never
    needs them."""
    backend, source = active()
    return {
        "url": backend.url,
        "label": backend.label,
        "source": source,
        "presets": [{"label": p.label, "url": p.url} for p in presets()],
    }


def _reset_for_tests() -> None:
    global _temporary, _saved, _saved_loaded
    _temporary, _saved, _saved_loaded = None, None, False
