#!/bin/bash
set -e

if [ "$EUID" -eq 0 ]; then
    echo "Warning: running as root may cause permission issues."
fi

if [ ! -d "env" ]; then
    echo "Please run 'run-install.sh' first to set up the environment."
    read -rp "Press enter to exit..." _
    exit 1
fi

printf "\033]0;Codename-RVC-Fork-4\007"
clear

export CPPFLAGS="-D_POSIX_C_SOURCE=200809L ${CPPFLAGS}"
export CFLAGS="-D_POSIX_C_SOURCE=200809L ${CFLAGS}"

env/bin/python app.py --open
