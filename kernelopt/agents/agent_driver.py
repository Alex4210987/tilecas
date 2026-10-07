"""One optimizer CLI behind one argv/event contract.

The launcher owns the evaluator, the walltime envelope and the evidence
record; only the optimizer's command line, its streamed event schema and the
location of its transcripts differ between vendors.  Codex stays the default
so existing ledgers replay unchanged, and ``KERNELBENCH_AGENT_DRIVER=claude``
selects Claude Code, whose events are normalized into the Codex
``thread.started``/``item.*`` shape so no caller has to branch.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import uuid

# Every difference between the optimizers is a row in this table, so adding one
# is an entry here rather than another branch somewhere else. A module that
# needs a driver's paths reads them from here; copying the table is how a
# dashboard and an isolation routine came to disagree about where a
# conversation is kept.
DRIVERS = ("codex", "claude")
DEFAULT_DRIVER = "codex"
DEFAULT_MODELS = {"codex": "gpt-6-astra", "claude": "opus"}
HOME_DIRS = {"codex": ".codex", "claude": ".claude"}
TRANSCRIPT_DIRS = {"codex": ".codex/sessions", "claude": ".claude/projects"}

NAME = os.environ.get("KERNELBENCH_AGENT_DRIVER", DEFAULT_DRIVER).lower()
if NAME not in DRIVERS:
    raise ValueError(f"Unsupported optimizer driver: {NAME}; expected one of {', '.join(DRIVERS)}")
CODEX = NAME == "codex"
DEFAULT_MODEL = DEFAULT_MODELS[NAME]

# Where this driver keeps configuration, credentials and conversations.
HOME_DIR = HOME_DIRS[NAME]
TRANSCRIPTS = TRANSCRIPT_DIRS[NAME]
# The controller-owned copy that replaces it inside an isolated workspace.
PRIVATE = ".harness/codex-sessions" if CODEX else ".harness/claude-sessions"
# Codex reuses the controller's rotating OAuth login. Claude Code sessions
# take a long-lived token instead (`claude setup-token`): a rotating refresh
# token has exactly one valid holder, so sharing one with an interactive
# session ends the run the moment either side rotates it.
CREDENTIALS = ("config.toml", "auth.json", "models.json") if CODEX else ("oauth-token", "settings.json")

# Claude Code takes its working directory from the process, and the Ascend
# identity wrapper execs through ``env``, which does not chdir. The same shim
# loads the long-lived token from a file rather than taking it as an argument,
# so it never appears in argv where any account on the host could read it.
_LAUNCH = ["sh", "-c",
           'cd "$1" || exit 1\n'
           'if [ -r "${CLAUDE_CONFIG_DIR:-}/oauth-token" ]; then\n'
           '  CLAUDE_CODE_OAUTH_TOKEN=$(cat "$CLAUDE_CONFIG_DIR/oauth-token")\n'
           '  export CLAUDE_CODE_OAUTH_TOKEN\n'
           'fi\n'
           'shift\n'
           'exec "$@"',
           "kernelbench-optimizer"]
# Tool names the cost ledger groups, by the Codex item type they map to.
_TOOL_KIND = {"Bash": "command_execution", "BashOutput": "command_execution",
              "KillShell": "command_execution", "WebSearch": "web_search",
              "WebFetch": "web_search", "Edit": "file_change",
              "Write": "file_change", "NotebookEdit": "file_change"}


# Output that means "ask again later" rather than "this run is broken".
# Codex says the model is at capacity; Claude Code reports a usage window.
_RETRYABLE = ("at capacity",) if CODEX else (
    "at capacity", "rate limit", "rate_limit", "usage limit", "overloaded", "429")
# A rotating capacity notice clears in seconds; a usage window does not.
BACKOFF_SECONDS = 1.0 if CODEX else 300.0


def retryable(output):
    """True when the optimizer stopped for capacity, not for a real fault."""
    text = (output or "").lower()
    return any(marker in text for marker in _RETRYABLE)


def new_session_id():
    """A controller-owned conversation id, or None when the CLI assigns it."""
    return None if CODEX else str(uuid.uuid4())


def executable():
    """A CLI path an unprivileged session identity can actually execute.

    A per-user install sits under a private home the optimizer's identity
    cannot traverse, so the deployment uses the system-wide copy.
    """
    if CODEX:
        candidates = (os.environ.get("KERNELBENCH_CODEX_BIN"),
                      "/usr/lib/node_modules/@openai/codex/bin/codex.js",
                      "/usr/local/lib/node_modules/@openai/codex/bin/codex.js",
                      shutil.which("codex"))
        for candidate in candidates:
            if candidate and os.access(candidate, os.X_OK):
                return candidate
        raise FileNotFoundError("No executable Codex CLI; set KERNELBENCH_CODEX_BIN")
    for candidate in (os.environ.get("KERNELBENCH_CLAUDE_BIN"),
                      "/usr/local/bin/claude", shutil.which("claude")):
        if candidate and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError("No executable Claude Code CLI; set KERNELBENCH_CLAUDE_BIN")


def credential_directory():
    """Select this driver's credentials without borrowing another driver's home."""
    if os.environ.get("KERNELBENCH_DRIVER_HOME"):
        return Path(os.environ["KERNELBENCH_DRIVER_HOME"]).expanduser() / HOME_DIR
    if CODEX:
        return Path(os.environ.get("CODEX_HOME", str(Path.home() / HOME_DIR))).expanduser()
    return Path("/var/lib/ako/agent-claude") / HOME_DIR


def home_environment(private):
    """Environment that points the optimizer at its own private state."""
    if CODEX:
        return {"CODEX_HOME": str(private)}
    return {"CLAUDE_CONFIG_DIR": str(private),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}


# How hard the optimizer works before it answers. It also decides how far it
# explores on its own, which is the variable a scaffolding comparison cares
# about: a lower setting makes fewer, more consolidated tool calls.
EFFORT = os.environ.get("KERNELBENCH_AGENT_EFFORT", "low" if CODEX else "high")


def build(model, prompt, view, *, json_output=False, resume=None,
          single_agent=False, session_id=None, skip_git_check=True):
    """Return argv for one optimizer invocation rooted at ``view``."""
    view = str(view)
    if CODEX:
        flags = (["--json"] if json_output else []) \
            + (["--skip-git-repo-check"] if skip_git_check else [])
        tuning = (["-c", "features.multi_agent=false"] if single_agent else []) \
            + ["-c", f"model_reasoning_effort={EFFORT}", "-m", model]
        if resume:
            # ``resume`` keeps the same rollout; its options follow the verb.
            return ["codex", "exec", "--sandbox", "danger-full-access", "-C", view,
                    "resume"] + flags + tuning + [resume, prompt]
        return ["codex", "exec"] + flags + ["--sandbox", "danger-full-access"] \
            + tuning + ["-C", view, prompt]

    # One non-interactive streamed turn on a known conversation id.
    # --disallowedTools is variadic, so it must never sit last: it would
    # swallow the prompt as another tool name.
    command = ["claude", "-p"]
    if single_agent:
        # No delegation: the paper's single-agent arms own every edit.
        command += ["--disallowedTools", "Task"]
    command += ["--model", model, "--effort", EFFORT, "--output-format", "stream-json",
                "--verbose", "--dangerously-skip-permissions"]
    command += ["--resume", resume] if resume else (["--session-id", session_id] if session_id else [])
    return _LAUNCH + [view] + command + [prompt]


class Stream:
    """Normalize a driver's stdout lines into Codex-shaped events."""

    def __init__(self):
        self.kinds = {}
        self.session_id = None

    def normalize(self, line):
        """Events for one line, or None when the line is not an event."""
        try:
            event = json.loads(line)
        except ValueError:
            return None
        if not isinstance(event, dict):
            return None
        if CODEX:
            if event.get("type") == "thread.started":
                self.session_id = event.get("thread_id") or self.session_id
            return [event]
        kind, message = event.get("type"), event.get("message") or {}
        if kind == "system" and event.get("subtype") == "init":
            self.session_id = event.get("session_id") or self.session_id
            return [{"type": "thread.started", "thread_id": event.get("session_id")}]
        if kind == "result":
            return [{"type": "turn.completed", "usage": event.get("usage", {})}]
        if kind == "rate_limit_event":
            info = event.get("rate_limit_info") or {}
            if info.get("status") not in (None, "allowed"):
                return [{"type": "item.completed",
                         "item": {"type": "agent_message", "id": str(uuid.uuid4()),
                                  "text": f"rate limit: {json.dumps(info)}"}}]
            return []
        if kind == "assistant":
            return [e for block in message.get("content") or [] for e in self._sent(block)]
        if kind == "user":
            return [self._returned(block) for block in message.get("content") or []
                    if block.get("type") == "tool_result"]
        return []

    def _sent(self, block):
        if block.get("type") == "text" and block.get("text", "").strip():
            return [{"type": "item.completed",
                     "item": {"type": "agent_message", "id": str(uuid.uuid4()), "text": block["text"]}}]
        # Why an edit was made is the part of a run that is hardest to
        # reconstruct afterwards, and it was being dropped: a block that is
        # neither text nor a tool call used to fall through silently.
        if block.get("type") == "thinking" and (block.get("thinking") or "").strip():
            return [{"type": "item.completed",
                     "item": {"type": "reasoning", "id": str(uuid.uuid4()), "text": block["thinking"]}}]
        if block.get("type") != "tool_use":
            return []
        name = block.get("name", "")
        kind = _TOOL_KIND.get(name, "mcp_tool_call" if name.startswith("mcp__") else "local_shell_call")
        self.kinds[block.get("id")] = kind
        payload = block.get("input") or {}
        command = payload.get("command") or payload.get("file_path") or json.dumps(payload)[:400]
        return [{"type": "item.started",
                 "item": {"type": kind, "id": block.get("id"), "command": f"{name}: {command}"}}]

    def _returned(self, block):
        identifier = block.get("tool_use_id")
        failed = bool(block.get("is_error"))
        content = block.get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return {"type": "item.completed",
                "item": {"type": self.kinds.pop(identifier, "local_shell_call"), "id": identifier,
                         "aggregated_output": content if isinstance(content, str) else "",
                         "status": "failed" if failed else "completed",
                         "exit_code": 1 if failed else 0}}


def render(line, stream=None):
    """Readable log text for one stdout line of any driver."""
    events = (stream or Stream()).normalize(line)
    if events is None:
        return line
    lines = []
    for event in events:
        item = event.get("item", {})
        if event.get("type") == "thread.started":
            lines.append(f"[session {event.get('thread_id')}]")
        elif event.get("type") in {"error", "turn.failed"}:
            lines.append(json.dumps(event, ensure_ascii=False))
        elif event.get("type") == "item.started":
            lines.append("$ " + item.get("command", ""))
        elif event.get("type") == "item.completed":
            text = (item.get("text") or item.get("aggregated_output") or "").rstrip()
            if not text:
                continue
            if item.get("type") == "reasoning":
                # Marked so a reader can follow it, or strip it with one grep.
                lines.extend("\u00b7 " + row for row in text.splitlines())
            else:
                lines.append(text)
    return "".join(line + "\n" for line in lines)
