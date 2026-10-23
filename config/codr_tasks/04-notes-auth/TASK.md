This continues task `03-notes-api`. Copy `../03-notes-api/app.py` and `../03-notes-api/test_app.py` into this folder and extend them here; leave the task-03 folder as it is. Same rules as before: Python 3's standard library only, nothing can be installed.

Add user accounts and schema migrations. Everything task 03 asked for still holds, except that the notes endpoints now need a login.

Users:
- `POST /users` with `{"username": "...", "password": "..."}` → `201` and `{"id": <int>, "username": "..."}`. A missing or empty field → `400`; a username that is taken → `409`. Every error response is JSON with an `"error"` key.
- `POST /login` with the same body → `200` and `{"token": "..."}`; a wrong username or password → `401`.
- Never store a password in plain text: keep a salted hash (`hashlib.pbkdf2_hmac` or `hashlib.scrypt`).

Notes:
- Every `/notes` endpoint needs the header `Authorization: Bearer <token>`. No header or a bad token → `401`.
- Notes belong to the user who created them. A user sees and changes only their own notes; another user's note id answers `404`, the same as an id that does not exist.
- `GET /` may stay public.

Migrations:
- Keep the database schema version in the database itself, and upgrade an older database on start, in place, without losing its notes.
- `app.py` must open a database written by the **task-03** `app.py` and upgrade it. Notes from before the upgrade go to a user `legacy`, created by the upgrade, whose password is the environment variable `NOTES_LEGACY_PASSWORD` (default `legacy`).
- Starting the upgraded app again on an upgraded database must change nothing.

Update `test_app.py` to cover the new behaviour, including a migration from a task-03 database, and make sure `python3 -m unittest` passes.
