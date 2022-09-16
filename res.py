#!/usr/bin/env python3

# This file is part of CorPipe <https://github.com/ufal/crac2022-corpipe>.
#
# Copyright 2022 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=str, help="Experiment name")
parser.add_argument("epochs", default=0, nargs="?", type=int, help="Epochs to show")
parser.add_argument("-c", default=None, type=str, help="Compare to another experiment")
args = parser.parse_args()

treebanks = ["ca", "cs_pced", "cs_pdt", "de_pot", "de_par", "en_par", "en_gum", "es", "fr", "hu", "lt", "pl", "ru"]
high_resource = ["ca", "cs_pced", "cs_pdt", "en_gum", "es", "fr", "hu", "pl", "ru"]

# Load the data
def load(exp):
    exp_name, exp_kind = os.path.splitext(exp)
    exp_suffix = {
        "": ".eval", ".0": ".0.eval", ".1": ".1.eval", ".10": ".10.eval", ".100": ".100.eval",
        ".s": ".evals", ".0s": ".0.evals", ".1s": ".1.evals", ".10s": ".10.evals", ".100s": ".100.evals",
        ".h": ".headsonly.eval", ".0h": ".0headsonly.eval", ".1h": ".1headsonly.eval", ".10h": ".10headsonly.eval", ".100h": ".100headsonly.eval",
        ".hs": ".headsonly.evals", ".0hs": ".0headsonly.evals", ".1hs": ".1headsonly.evals", ".10hs": ".10headsonly.evals", ".100hs": ".100headsonly.evals",
    }[exp_kind]
    results = {}
    for path in sorted(glob.glob(f"logs/{exp_name}*/*[0-9]{exp_suffix}")):
        base, epoch, *_ = os.path.basename(path)[:-len(exp_suffix)].split(".")
        for treebank in treebanks:
            if base.startswith(treebank):
                base = treebank
        if base not in treebanks:
            raise ValueError(f"Unknown treebank for evaluation '{base}'")
        results.setdefault(base, {})
        if epoch in results[base]:
            raise ValueError(f"Multiple evaluations for '{base}' epoch '{epoch}'")
        with open(path, "r", encoding="utf-8") as eval_file:
            for line in eval_file:
                line = line.rstrip("\r\n")
                if line.startswith("CoNLL score: "):
                    results[base][epoch] = line[13:]
    return results
results = load(args.exp)

# Print them out
def avg(callback, results):
    values = [callback(results[t]) if t in results else "" for t in treebanks]
    if all(value for t, value in zip(treebanks, values) if t in high_resource):
        values.append("{:.2f}".format(sum(float(value) for t, value in zip(treebanks, values) if t in high_resource) / (len(high_resource))))
    if all(values):
        values.append("{:.2f}".format(sum(float(value) for value in values) / len(values)))
    return values
if args.c:
    others = load(args.c)
    def show(callback):
        xs, ys = avg(callback, results), avg(callback, others)
        return ["\033[{}m{:+.2f}\033[0m".format(32 if float(x) >= float(y) else 31,
                                                float(x) - float(y)) if x and y else ""
                for x, y in zip(xs, ys)]
else:
    show = lambda callback: avg(callback, results)
print("mode", *treebanks, "avg-hig", "avg", sep="\t")
print("last", *show(lambda res: list(res.values())[-1]), sep="\t")
print("best-10", *show(lambda res: max(list(res.values())[-10:], key=float)), sep="\t")
print("best", *show(lambda res: max(list(res.values()), key=float)), sep="\t")
for epoch in range(args.epochs):
    print(epoch, *show(lambda res: list(res.values())[epoch] if len(res) > epoch else ""), sep="\t")
