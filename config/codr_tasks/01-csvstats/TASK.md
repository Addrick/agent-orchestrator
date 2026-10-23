Build `csvstats.py`, a command-line tool that summarizes the numeric columns of a CSV file.

Use Python 3 with the standard library only. The sandbox has no internet, so nothing can be installed.

Usage: `python3 csvstats.py FILE [--columns a,b,...] [--format table|json]`

- The first row of the file is the header.
- A column is numeric when every non-empty cell in it parses as a number. Empty cells are ignored. Columns that are not numeric are skipped, unless they are named in `--columns`.
- For each numeric column, report `count` (non-empty cells), `mean`, `min`, `max` and `stdev` (the sample standard deviation; `null` when count < 2).
- `--columns` limits the output to the named columns, in that order. A named column that is missing or not numeric is an error: print a message to stderr and exit with status 2.
- `--format json` (the default is `table`) prints exactly one JSON object mapping each column name to `{"count", "mean", "min", "max", "stdev"}`, with columns in file order (or `--columns` order). `table` is a human-readable table; its layout is up to you.
- `--help` describes all of this.
- A file that does not exist is an error: exit status 2.

Write tests with `unittest` in `test_csvstats.py` and make sure `python3 -m unittest` passes.
