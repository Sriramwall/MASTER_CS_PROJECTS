#!/usr/bin/env python
"""mapper.py"""

import sys

# input comes from STDIN (standard input)

top_words = ['nba','game','team','part','sport','turner','play','player','point','last']

for line in sys.stdin:
    line = line.strip()
    words = line.split()
    for word in words:
        if word in top_words:
            for w in words:
                if w != word:
                    print('<%s,%s>\t%s' % (word, w, 1))
