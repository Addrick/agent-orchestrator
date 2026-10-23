Build `ssg.py`, a static site generator that turns a folder of Markdown posts into HTML. Python 3's standard library only; nothing can be installed (no `markdown` package: write the converter yourself). This is a bigger task than the earlier ones, so plan it, build it in pieces, and test each piece.

Usage: `python3 ssg.py SRC OUT`. It builds `SRC` into `OUT`, creating `OUT` if needed, and exits 0. An error (no `SRC/content`, a template missing, a post without a `title`) prints a message on stderr and exits with status 1.

**Input layout**
- `SRC/content/*.md`: the posts. The file name without `.md` is the post's *slug*.
- `SRC/templates/post.html` and `SRC/templates/index.html`: the templates.

**Front matter.** A post may start with a block between two lines that are exactly `---`. Each line in it is `key: value` (split at the first `:`, both sides stripped). `title` is required. `date` is optional, as `YYYY-MM-DD`. Other keys are allowed and are available to templates.

**Templates.** `{{ name }}` (spaces inside the braces optional) is replaced by the value of `name`. Values from front matter are HTML-escaped when inserted; `content` and `posts` are inserted as-is. An unknown name becomes the empty string.
- `post.html` gets every front-matter key, plus `slug` and `content` (the post's HTML).
- `index.html` gets `posts`: one `<li><a href="SLUG.html">TITLE</a></li>` per post (title HTML-escaped), newest `date` first. Posts without a date come after the dated ones. Ties are ordered by slug, ascending. Everything is joined with `\n`.

**Output.** `OUT/SLUG.html` for each post, and `OUT/index.html`. Write files as UTF-8.

**Markdown subset.** Convert exactly this, and HTML-escape `&`, `<` and `>` in all text:
- Blocks are separated by one or more blank lines.
- `# ` to `###### ` at the start of a line: a heading, `<h1>…</h1>` to `<h6>…</h6>`.
- A fenced code block: a line that is exactly ```` ``` ````, optionally followed by a language name, up to the next line that is exactly ```` ``` ````. Output `<pre><code>…</code></pre>` with the lines kept verbatim (only escaped, no inline formatting). The block may contain blank lines.
- A block whose lines all start with `- `: `<ul><li>…</li>…</ul>`, one item per line.
- A block whose lines all start with a number, a `.` and a space (`1. `): `<ol><li>…</li>…</ol>`.
- Any other block: a paragraph, `<p>…</p>`, its lines joined with a single space.
- Inline, inside headings, list items and paragraphs: `` `code` `` → `<code>code</code>` (no other formatting inside it), `**bold**` → `<strong>bold</strong>`, `*em*` → `<em>em</em>`, and `[text](url)` → `<a href="url">text</a>`.

**Incremental builds.** A second build of an unchanged `SRC` into the same `OUT` must not rewrite any file. When one post changes, only that post's page and `index.html` are rewritten. When a template changes, every page that uses it is rebuilt. When a post is deleted, its page is removed from `OUT`. Each build prints one last line exactly like `built 2, skipped 5, removed 0` (pages written, pages left alone, pages removed; `index.html` counts as a page).

Write tests with `unittest` in `test_ssg.py` and make sure `python3 -m unittest` passes.
