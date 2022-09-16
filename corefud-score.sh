#!/bin/sh

# This file is part of CorPipe <https://github.com/ufal/crac2022-corpipe>.
#
# Copyright 2022 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

[ $# -ge 2 ] || { echo Usage: $0 gold_file system_file >&2; exit 1; }

R=$(dirname $0)

lang="$(basename "$2")"
lang="${lang%%_*}"
($R/venv/bin/python3 $R/corefud-scorer/corefud-scorer.py "$1" "$2" -m muc bcub ceafe
 $R/venv/bin/python3 $R/validator/validate.py --level 2 --coref --lang $lang "$2") >"${2%.conllu}.eval" 2>&1
$R/venv/bin/python3 $R/corefud-scorer/corefud-scorer.py "$1" "$2" -m muc bcub ceafe -s >"${2%.conllu}.evals" 2>&1
