# src/interfaces/portal_render.py

"""Serve kobold-lite's portal.html with the DERPR portal type injected.

The portal page enables its DERPR layer (engine routing, persona models, OAI
endpoint auto-connect) based on `window.DERPR_PORTAL_TYPE`. Historically that was
sniffed client-side from `window.location.port` (5003=engine, 5002=passthrough),
which breaks whenever the page is reached over an SSH tunnel, reverse proxy, or
domain where the browser's URL port isn't the adapter's port.

Instead, the adapter that serves the page declares the type server-side: we inject
`window.DERPR_PORTAL_TYPE_INJECTED` just before the page's init-config script. The
init script honours the injected value first and only falls back to port sniffing
when the page is opened directly without injection.
"""

import os
from typing import Dict

_PORTAL_PATH = os.path.join(os.path.dirname(__file__), "web_assets", "portal.html")
_INJECT_ANCHOR = '<script id="init-config">'

_cache: Dict[str, str] = {}


def render_portal_html(portal_type: str) -> str:
    """Return portal.html with `window.DERPR_PORTAL_TYPE_INJECTED` set.

    `portal_type` is a trusted server-side literal ("engine"/"passthrough"),
    never user input, so embedding it in a script tag is safe. Result is cached
    per type for the process lifetime.
    """
    cached = _cache.get(portal_type)
    if cached is not None:
        return cached

    with open(_PORTAL_PATH, encoding="utf-8") as f:
        html = f.read()

    inject = f'<script>window.DERPR_PORTAL_TYPE_INJECTED="{portal_type}";</script>\n\t'
    rendered = html.replace(_INJECT_ANCHOR, inject + _INJECT_ANCHOR, 1)
    _cache[portal_type] = rendered
    return rendered
