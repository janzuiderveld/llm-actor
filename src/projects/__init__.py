"""
Project bootstrap utilities for configuring and orchestrating Pipecat Thin runtimes.

Individual projects can import from :mod:`projects.utils` to reuse the helpers that
reset runtime state, apply config overrides, and spawn background processes.
"""

from .utils import (
    ACTIONS_FILE,
    INBOX_FILE,
    PARAMS_FILE,
    REPO_ROOT,
    LockFile,
    ProjectConfig,
    acquire_process_lock,
    apply_runtime_config_overrides,
    append_action,
    append_inbox_line,
    append_json_line,
    ensure_runtime_state,
    ensure_runtime_files,
    launch_module,
    load_project_config,
    python_module_args,
    reset_runtime_state,
    set_system_prompt,
    tail_line,
    spawn_subprocess,
    terminate_processes,
)

__all__ = [
    "ACTIONS_FILE",
    "INBOX_FILE",
    "PARAMS_FILE",
    "REPO_ROOT",
    "LockFile",
    "ProjectConfig",
    "acquire_process_lock",
    "apply_runtime_config_overrides",
    "append_action",
    "append_inbox_line",
    "append_json_line",
    "ensure_runtime_state",
    "ensure_runtime_files",
    "launch_module",
    "load_project_config",
    "python_module_args",
    "reset_runtime_state",
    "set_system_prompt",
    "tail_line",
    "spawn_subprocess",
    "terminate_processes",
]
