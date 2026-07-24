#!/bin/bash
# Detect --pc2 / --fir in script arguments and export C2HLS_SITE.
# Usage: source scripts/bootstrap_site.sh "$@"

for _arg in "$@"; do
  case "$_arg" in
    --pc2) export C2HLS_SITE=pc2; break ;;
    --fir) export C2HLS_SITE=fir; break ;;
  esac
done
