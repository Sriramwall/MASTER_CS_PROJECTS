#!/usr/bin/env python
"""mapper.py"""

import sys

# input comes from STDIN (standard input)
top_words = ['sport','illustr','favorit','sign','pleas','servic','custom','team','point','list']

for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()
    # split the line into words
    words = line.split()
    # increase counters
    for word in words:
        # write the results to STDOUT (standard output);
        # what we output here will be the input for the
        # Reduce step, i.e. the input for reducer.py
        #
        # tab-delimited; the trivial word count is
        if word in top_words:
            for w in words:
                if w != word:
                    print('<%s,%s>\t%s' % (word, w, 1))
