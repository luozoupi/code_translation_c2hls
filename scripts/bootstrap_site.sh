#!/bin/bash
# Detect --pc2 in script arguments and export C2HLS_SITE=pc2.
# Usage: source scripts/bootstrap_site.sh "$@"

for _arg in "$@"; do
  if [[ "$_arg" == "--pc2" ]]; then
    export C2HLS_SITE=pc2
    break
  fi
done
