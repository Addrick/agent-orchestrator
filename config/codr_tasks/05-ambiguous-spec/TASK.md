Build `wordfreq.py`, a command-line tool that reports the most common words in a text. Python 3's standard library only; nothing can be installed.

Usage: `python3 wordfreq.py [FILE ...] [-n N] [--json]`

- Read the named files, or standard input when no file is given.
- Print the `N` most common words (default 10), most common first, one per line as `word count`.
- `--json` prints one JSON list of `[word, count]` pairs instead, in the same order.
- Words are compared case-insensitively and printed in lower case.
- A file that does not exist is an error: a message on stderr and exit status 2.

Some details are not specified on purpose: what counts as a word, how ties are ordered, and anything else you find. Do not wait for an answer. Pick a reasonable rule for each, follow it consistently, and write each choice and why you made it in `DECISIONS.md`.

Write tests with `unittest` in `test_wordfreq.py` and make sure `python3 -m unittest` passes.
