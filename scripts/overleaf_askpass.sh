#!/bin/bash
# Git askpass helper for the Overleaf remote: username "git", password = token from .env.
case "$1" in
  Username*) echo git ;;
  *) grep '^OVERLEAF_GIT_TOKEN=' "$(dirname "$0")/../.env" | cut -d= -f2- ;;
esac
