#!/usr/bin/env bash
# Run an audit script with the devenv C++ runtime and offline Hugging Face access.
cd "$(dirname "$0")"
exec direnv exec /home/rajveer/Projects/personal/capstone-mmd-audit env HF_HUB_OFFLINE=1 .venv/bin/python "$@" 2> >(grep -v -E "^(•|✓|Resolved|Checked)" >&2)
