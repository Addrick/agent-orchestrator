Build `todo.py`, a small command-line todo list. Use the `click` library for the command-line interface.

Usage: `python3 todo.py [--file PATH] COMMAND ...`. The list lives in the JSON file at `PATH` (default `todo.json` in the current directory), created on first use.

- `add TEXT...` adds an item (the words joined with spaces) and prints `added N`, where `N` is its id. Ids start at 1 and are never reused, even after a removal.
- `list` prints every item in id order, one per line, as the id, `[ ]` or `[x]`, and the text (`3 [x] buy milk`). `list --pending` shows only the items not done.
- `done N` marks item `N` done; `undo N` marks it not done.
- `remove N` deletes item `N`.
- An id that does not exist: a message on stderr and exit status 1. Every other success exits 0.

Write tests with `unittest` in `test_todo.py` and make sure `python3 -m unittest` passes.
