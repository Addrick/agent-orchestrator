Build `cronlib`, a Python library for classic five-field cron expressions, as a package: `cronlib/__init__.py` (more modules inside it are fine). Python 3's standard library only; nothing can be installed, and there is no `croniter` to copy from. This is a library built from a written spec alone, so read the whole spec before you start.

**API**
- `cronlib.parse(expr)` returns a `Cron`. Any invalid expression raises `cronlib.CronError`, which is a subclass of `ValueError`, with a message that says what is wrong.
- `Cron.minute`, `.hour`, `.day`, `.month`, `.weekday`: the sorted tuple of the ints that field allows. Weekdays are 0-6 with Sunday = 0 (a `7` in an expression is also Sunday).
- `Cron.matches(dt)`: whether the naive `datetime` `dt` falls in a minute the expression fires in (its seconds and microseconds are ignored).
- `Cron.next_fire(after)`: the earliest fire time strictly after `after`, as a `datetime` with second and microsecond 0.
- `Cron.prev_fire(before)`: the latest fire time strictly before `before`, the same way.
- `Cron.iter(after)`: a generator of successive fire times after `after`.
- An expression that can never fire (such as `0 0 31 2 *`) makes `next_fire` and `prev_fire` raise `CronError`. They must never loop forever, and must stay fast for rare schedules: `0 0 29 2 *` searched from 2097-03-01 must find 2104-02-29 in well under a second.

**Syntax**
- Exactly five fields separated by whitespace: minute (0-59), hour (0-23), day of month (1-31), month (1-12), day of week (0-7).
- A field is a comma-separated list of items. An item is `*`, `N`, or `A-B` (A <= B, no wraparound), optionally followed by a step `/S` with S >= 1: `*/S`, `A-B/S`, and `N/S`, which means `N-max/S` (for example `5/20` in the minute field is 5, 25, 45).
- Month names `JAN`-`DEC` and day names `SUN`-`SAT`, in any case, can stand wherever a number can in their own field, ranges included (`MON-FRI`). Nowhere else.
- Anything else is an error: a wrong number of fields, a value out of range, a backwards range, an empty item, step 0, unknown names, stray characters.
- Macros, as the whole expression, in any case: `@yearly` and `@annually` = `0 0 1 1 *`, `@monthly` = `0 0 1 * *`, `@weekly` = `0 0 * * 0`, `@daily` and `@midnight` = `0 0 * * *`, `@hourly` = `0 * * * *`. Any other `@` word is an error.

**The day rule** (as in Vixie cron). A field is *restricted* when its text does not start with `*`. If both day of month and day of week are restricted, a day matches when **either** matches: `0 0 13 * 5` fires on every 13th and on every Friday. Otherwise both must match, and an unrestricted one matches every day, so `0 0 */2 * 1` fires only on Mondays that fall on an odd-numbered day.

A hidden test suite grades the library with a few hundred cases, in groups. When it fails you see only how many cases in each group passed, not which ones, so write your own tests from the spec: put them in `test_cronlib.py` with `unittest`, and make sure `python3 -m unittest` passes.
