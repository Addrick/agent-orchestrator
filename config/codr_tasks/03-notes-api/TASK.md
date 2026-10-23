Build a small notes web service in `app.py`, with Python 3's standard library only (`http.server`, `sqlite3`, `json`). The sandbox has no internet, so nothing can be installed.

`python3 app.py --port PORT [--db PATH]` serves on 127.0.0.1:PORT until it is killed. Notes persist in the SQLite file at PATH (default `notes.db` next to `app.py`), so they survive a restart.

JSON API (request and response bodies are JSON):
- `POST /notes` with `{"text": "..."}` → `201` and the note `{"id": <int>, "text": "...", "created": "<ISO 8601 time>"}`. A missing or empty `text`, or a body that is not JSON, → `400` with `{"error": "..."}`.
- `GET /notes` → `200` and a list of all notes, oldest first. `GET /notes?q=word` returns only notes whose text contains `word` (case-insensitive).
- `GET /notes/<id>` → `200` and that note; an unknown id → `404` with `{"error": "..."}`.
- `PUT /notes/<id>` with `{"text": "..."}` → `200` and the updated note (same id and `created`); unknown id → `404`; bad body → `400`.
- `DELETE /notes/<id>` → `204` with no body; unknown id → `404`.
- `GET /` → `200`, an HTML page that lists the notes and has a form to add one.

Write tests with `unittest` in `test_app.py` that start the server on a free port and talk to it over HTTP, and make sure `python3 -m unittest` passes. Run the server yourself and try it with `curl` before you say you are done.
