"""Language-model backends. All return plain text; the caller parses and validates JSON.

- "claude-cli": the local `claude -p` command (Claude Code), no API key handled by this repo.
- "anthropic": the Anthropic Messages API; reads ANTHROPIC_API_KEY from the environment.
Choose with the RAG_BACKEND environment variable (default: claude-cli).
"""
import json
import os
import shutil
import subprocess

MODEL = os.environ.get("RAG_MODEL", "haiku")  # alias works across Claude Code versions


def complete(prompt, system, backend=None, schema=None, timeout=180):
    """Return the model reply: a dict when the backend enforces `schema`, otherwise text."""
    backend = backend or os.environ.get("RAG_BACKEND", "claude-cli")
    if backend == "claude-cli":
        return _claude_cli(prompt, system, schema, timeout)
    if backend == "anthropic":
        return _anthropic(prompt, system)
    raise ValueError(f"unknown backend {backend!r}")


def _claude_cli(prompt, system, schema, timeout):
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("`claude` command not found; install Claude Code or set RAG_BACKEND=anthropic")
    # prompt goes in on stdin so long passages never hit argv limits; no tools, no saved session
    cmd = [exe, "-p", "--model", MODEL, "--system-prompt", system, "--output-format", "json",
           "--tools", "", "--no-session-persistence"]
    if schema:
        cmd += ["--json-schema", json.dumps(schema)]
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout)
    envelope = json.loads(r.stdout) if r.stdout.strip() else {}
    if r.returncode != 0 or envelope.get("is_error") or not envelope:
        detail = envelope.get("result") or r.stderr.strip()
        raise RuntimeError(f"claude -p failed (rc={r.returncode}): {detail[:300]}")
    return envelope.get("structured_output") or envelope["result"]


def _anthropic(prompt, system):
    import anthropic  # optional dependency
    msg = anthropic.Anthropic().messages.create(
        model=os.environ.get("RAG_MODEL", "claude-haiku-5-5"), max_tokens=1024, system=system,
        messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if b.type == "text")
