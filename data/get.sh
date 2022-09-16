#!/bin/sh

# This file is part of CorPipe <https://github.com/ufal/crac2022-corpipe>.
#
# Copyright 2022 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# Train and dev
wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4698/CorefUD-1.0-public.zip
unzip CorefUD-1.0-public.zip
for f in CorefUD-1.0-public/data/*/*.conllu; do
  lang=$(basename $f)
  lang=${lang%%-*}
  mkdir -p $lang
  mv $f $lang/$(basename $f)
done
rm -r CorefUD-1.0-public/ CorefUD-1.0-public.zip

# Test
mkdir test
(cd test
 wget https://ufal.mff.cuni.cz/~popel/corefud-1.0/test-blind.zip
 unzip test-blind.zip
 for f in *.conllu; do
   lang=${f%%-*}
   mv $f ../$lang/$f
 done
)
rm -r test/

# Data cleanup
sed 's/20.1	trabajó	trabajar/20.1	_	_/' -i es_ancora/es_ancora-corefud-train.conllu
