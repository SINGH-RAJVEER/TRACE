#!/usr/bin/env bash
set -euo pipefail
exec devenv shell -- ssh -T \
	-o ControlMaster=auto \
	-o ControlPath=/home/rajveer/Projects/personal/capstone/artifacts/colab-teacher/ssh-control \
	-o BatchMode=yes -o ConnectTimeout=30 \
	-o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
	-o StrictHostKeyChecking=accept-new \
	-o UserKnownHostsFile=/home/rajveer/.ssh/known_hosts_colab_teacher \
	-i /home/rajveer/.ssh/colab_teacher_ed25519 \
	-o 'ProxyCommand=/home/rajveer/.local/bin/colab ssh --proxy-mode -s %h -i /home/rajveer/.ssh/colab_teacher_ed25519' \
	root@colab "$@"
