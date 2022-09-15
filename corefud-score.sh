#!/bin/sh

[ $# -ge 2 ] || { echo Usage: $0 gold_file system_file >&2; exit 1; }

R=$(dirname $0)

lang="$(basename "$2")"
lang="${lang%%_*}"
($R/venv/bin/python3 $R/corefud-scorer/corefud-scorer.py "$1" "$2" -m muc bcub ceafe
 $R/venv/bin/python3 $R/validator/validate.py --level 2 --coref --lang $lang "$2") >"${2%.conllu}.eval" 2>&1
$R/venv/bin/python3 $R/corefud-scorer/corefud-scorer.py "$1" "$2" -m muc bcub ceafe -s >"${2%.conllu}.evals" 2>&1
