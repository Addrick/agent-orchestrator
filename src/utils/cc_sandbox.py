# src/utils/cc_sandbox.py
"""The one builder for the `claude --settings` sandbox block (DP-314).

`docs/capability_map.md` recorded `providers/cc.py:build_cc_sandbox_settings()`
and `dispatcher.py:_sandbox_settings()` as independent copies at 0.68 vocabulary
similarity, verdict `unreviewed`, flagged as "a security-relevant divergence
risk". DP-314 needed to add `filesystem.allowWrite` to the sandbox — which would
have made it a divergence in *three* places, one of them the thing that decides
what an autonomous agent may overwrite. So the copies collapse here first and
both callers delegate.

The two call sites differ in exactly two ways, both expressed as arguments:
a capable dispatch adds the MCP bridge host to `allowedDomains`, and any caller
that linked the notes tree into its workspace adds that tree to `allowWrite`.
Everything else is common policy and now has one definition.

Lives in `utils` because both `src.engine` and `src.self_edit` must reach it and
`utils` is the dependency leaf (setup.cfg import-linter contracts).
"""

from typing import Any, Dict, Iterable, List, Optional

from config import global_config


def build_sandbox_settings(
    *,
    extra_domains: Optional[Iterable[str]] = None,
    allow_write: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Build the `--settings` payload, or None when CC_SANDBOX is off (in which
    case the flag is omitted entirely rather than passed empty).

    `autoAllowBashIfSandboxed` is on so a headless run never blocks on a prompt
    it cannot answer; the OS sandbox is what bounds it instead.

    `allow_write` grants OS-level write access to paths outside the working
    directory. Only pass paths that must be written — every entry widens the
    boundary that `--dangerously-skip-permissions` leans on.
    """
    if not global_config.CC_SANDBOX:
        return None

    sandbox: Dict[str, Any] = {"enabled": True, "autoAllowBashIfSandboxed": True}
    if global_config.CC_SANDBOX_WEAKER_NESTED:
        sandbox["enableWeakerNestedSandbox"] = True

    domains: List[str] = list(global_config.CC_SANDBOX_ALLOWED_DOMAINS)
    for domain in extra_domains or ():
        if domain and domain not in domains:
            domains.append(domain)
    if domains:
        sandbox["network"] = {"allowedDomains": domains}

    writes: List[str] = []
    for path in allow_write or ():
        if path and path not in writes:
            writes.append(path)
    if writes:
        sandbox["filesystem"] = {"allowWrite": writes}

    return {"sandbox": sandbox}
