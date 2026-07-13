#!/usr/bin/env bash
# .build_docs.sh -- autogenerate the docs site with pdoc: themed (One Dark), self-contained, no
# sphinx/config files. Ported from robosim's docs/build_docs.sh. microecs's shape:
#   * ONE API surface is documented: microecs/ (the ECS library, a real package). Any module that
#     fails to import is dropped (printed at the end), so a missing optional dep never breaks the build.
#   * The hand-written prose under docs/source/*.md is kept ON DISK AS-IS and rendered ALONGSIDE the
#     API: each becomes its own page under a synthetic "guide" package (its docstring = the markdown),
#     so prose and generated API share one themed sidebar. README.md is the home page (guide.html).
#   * index.html redirects to guide.html. resources/ images referenced by the prose are copied in.
#
#   ./.build_docs.sh            # build into ./public, then print a file:// link
#   ./.build_docs.sh public     # build into ./public  (GitLab Pages serves this dir in CI)
set -euo pipefail
cd "$(dirname "$0")"                    # repo root (this script lives at the microecs root)
OUT="${1:-public}"

# microecs/ is imported from the repo root.
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# one pure-python build dependency; auto-install so a fresh checkout just works
python -c "import pdoc" 2>/dev/null || { echo "[docs] installing pdoc..."; pip install --quiet pdoc; }

# --- embedded theme + template -> temp dir (pdoc prefers these files over its built-in defaults) ---
TMPL="$(mktemp -d)"
trap 'rm -rf "$TMPL"' EXIT

cat > "$TMPL/theme.css" <<'THEME_CSS'
:root {
  --pdoc-background: #282C34;
}
.pdoc {
  --text: white;
  --muted: whitesmoke;
  --link: var(--lightblue);
  --link-hover: white;
  --active: #555;
  --code: #232627;
  --accent: #232627;
  /* pdoc vars that onedark's theme.css omits -- map onto the palette so nothing falls back to plain text */
  --annotation: var(--orange);
  --def: var(--red);
  --name: var(--purple);
  --nav-hover: var(--dark);
  /* Actual theme colors */
  --lightblue: #61AFEF;
  --red: #E06C75;
  --green: #98C379;
  --dark: #282C34;
  --orange: #E5C07B;
  --purple: #B392F0;
  --blue: #9ECBFF;
  --silver: #ABB2BF;
}
THEME_CSS

cat > "$TMPL/custom.css" <<'CUSTOM_CSS'
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
}

.pdoc .il {
  color: #131416;
}

nav.pdoc a.pdoc-button.module-list-button {
  display: none;
}

.pdoc .modulename a:hover {
  color: whitesmoke;
}

.pdoc .classattr {
  color: #fff;
}

.pdoc .docstring .pdoc-code {
  background-color: #232627;
}

.pdoc pre {
  width: fit-content;
}

.pdoc pre a {
  color: whitesmoke;
}

.pdoc pre a:hover {
  color: #676767;
}


/* Submodules and API Doc. */

.pdoc h2 {
  color: #fff;
  font-weight: 0;
  margin: 0.3em 0;
  padding: 0.2em 0;
  content: "Module Content";
}

/* header */
.pdoc h5 {
  color: white;
  font-style: italic;
}

/* Module name */

.pdoc .modulename {
  color: #fff;
  font-weight: 0;
}

.pdoc-code .ow {
  color: var(--red);
}

nav.pdoc a.function {
  color: var(--purple);
}

nav.pdoc a.function:hover {
  background-color: var(--dark);
}

nav.pdoc .function::after {
  content: "";
}

nav.pdoc .function::before {
  content: "fn ";
  color: var(--red);
}

nav.pdoc a.class {
  color: whitesmoke;
}

/* Before the class name. This adds "class 'className'". */
nav.pdoc .class::before {
  content: "class ";
  color: var(--red);
}

/* After the class name. This will remove `()` That comes after the class name. */
nav.pdoc .class::after {
  content: "";
}

nav.pdoc a.class:hover {
  color: #fff;
  background-color: var(--dark);
}

nav.pdoc a.variable {
  color: whitesmoke;
}

/* Adds "var varname" to class variables. */
nav.pdoc a.variable::before {
  content: "var ";
  color: #9d9d9d;
  color: var(--red);
}

nav.pdoc a.variable:hover {
  color: #fff;
  background-color: var(--dark);
}

/* nav modules */
nav.pdoc li a {
  color: var(--purple);
}

nav.pdoc li a:hover {
  background-color: var(--dark);
}

/* Assigned values */
.pdoc span.default_value {
  color: var(--lightblue);
}

.pdoc span.def {
  font-weight: normal;
  color: var(--red);
}

/*
Attributes
----------
*/
.pdoc h6#attributes {
  color: #fff;
  font-size: 25px;
}

/*
Example
-------
*/

.pdoc h6#example {
  color: #fff;
  font-size: 29px;
}

/*
Notes
-----
*/

.pdoc h6#notes {
  font-size: 27px;
  color: var(--orange);
}

/*
Returns
-------
*/

.pdoc h6#returns {
  font-size: 27px;
  color: white;
}

/*
Parameters
----------
*/

.pdoc h6#parameters {
  font-size: 27px;
  color: white;
}

/*
Raises
------
*/

.pdoc h6#raises {
  font-size: 27px;
  color: rgb(218, 55, 55);
}

/* False, True, None */

.pdoc .kc {
  color: var(--lightblue);
}

.pdoc li {
  color: white;
}

/* Decorator module */

.pdoc .nd {
  color: var(--lightblue);
}

.pdoc b,
strong {
  color: rgba(255, 255, 255, 0.8);
  font-weight: normal;
}

/* markdown *emphasis* -- a plain italic that inherits the body color/size so it blends into the prose
   instead of shouting. (Admonitions are styled separately, via .pdoc-alert-note / -warning below.) */

.pdoc em {
  font-style: italic;
}

/* Decorator color */

.pdoc div.decorator {
  color: var(--lightblue);
}

div.pdoc-code.codehilite {
  background-color: var(--code);
}

/* class / function names */

.pdoc span.name {
  color: var(--purple);
  font-weight: normal;
}

/* Inherited class name color */

.pdoc span.base {
  color: whitesmoke;
}

/* Before inherited members colors, i.e., "builtins." */

.pdoc .inherited dt,
.pdoc .inherited dt::before {
  color: var(--silver);
}


/* Commas that separates parameters "," color */
.pdoc .inherited dd:not(:last-child)::after {
  color: #fff;
}

/* Commas that separates parameters "," color */

.pdoc .inherited dd:not(:last-child)::after {
  color: #fff;
}

/* Contents, Submodules, API Documentations */
/* This also can be separated */
.pdoc h1,
.pdoc h2,
.pdoc h3 {
  font-weight: 300;
  margin: 0.3em 0;
  padding: 0.2em 0;
  color: white;
}

/* Top left nav button */
nav.pdoc .module-list-button {
  display: inline-flex;
  align-items: center;
  margin-bottom: 1rem;
  color: white;
  border-color: white;
}

nav.pdoc .module-list-button:hover {
  border-color: white;
  color: white;
}

.pdoc pdoc-alert pdoc-alert-warning .p {
  color: black;
}

.pdoc-code {
  color: white;
}

.pdoc .pdoc-alert-warning {
  color: black;
  background-color: #d5a142;
  border-color: black;
}

.pdoc .pdoc-alert-note {
  color: rgb(7, 19, 24);
  background-color: rgb(184, 231, 251) ;
  border-color: var(--dark);
}

/* The nav bar */

nav.pdoc {
  background-color: rgb(55, 58, 72);
}

.pdoc-code .k {
  color: var(--red);
}

/* === local overrides (microecs) =================================================== */
/* widen the content column (layout.css ships 54rem) by 1.5x so 120-char-wide code
   examples render on a single line, without horizontal scrolling */
main, header { width: calc(81rem + var(--sidebar-width)); }
/* a Home link at the top of the sidebar, above search; styled like the "Modules"/"Contents" headings */
nav.pdoc a.home-button { display: inline-block; color: #fff; font-weight: 300; font-size: 2rem; margin: 0.3em 0; }
nav.pdoc a.home-button:hover { color: var(--lightblue); }
/* README-heading primitives stay code in the body, but render as plain text in the sidebar menu */
nav.pdoc code { font-family: inherit; background: none; color: inherit; padding: 0; border: 0; font-size: inherit; }
/* prose/README images must never overflow the content column */
.pdoc .docstring img { max-width: 100%; height: auto; }
/* prose tables (e.g. the Benchmarks page): pdoc's default has no visible borders. Make them
   full-width + "schematic" (crisp grid), with a header band and a subtle per-row hover shade. */
.pdoc .docstring table {
  display: table; width: 100%; max-width: 100%; table-layout: auto;  /* fully override pdoc's
  block/max-content/overflow default so the table fills the content column instead of shrink-wrapping */
  overflow: visible; border-collapse: collapse; margin: 1rem 0;
}
.pdoc .docstring th, .pdoc .docstring td {
  border: 1px solid rgba(255, 255, 255, 0.22); padding: 0.4rem 0.6rem; text-align: left; vertical-align: top;
}
.pdoc .docstring thead th { background: var(--accent); font-weight: 600; }
.pdoc .docstring tbody tr:hover { background: rgba(255, 255, 255, 0.06); }
/* Collapsible sidebar trees (Modules + the Examples group under Pages). Nodes that have
   children are <details><summary>label</summary><ul>kids</ul></details>: collapsed by default, so
   only the top level (microecs) shows; opening a node reveals its children, to any depth.
   The path to the CURRENT page is emitted <details open>, so you always see where you are. Both
   trees carry class="nav-tree". */
nav.pdoc > div > ul.nav-tree { margin-left: 0; }               /* cancel pdoc's negative root margin */
/* the disclosure row: hide the browser's default triangle, draw our own ▸ that rotates to ▾ on open */
nav.pdoc .nav-tree summary { list-style: none; cursor: pointer; }
nav.pdoc .nav-tree summary::-webkit-details-marker { display: none; }
nav.pdoc .nav-tree summary::before {
  content: "\25B8"; display: inline-block; width: 1rem; color: var(--silver); transition: transform 100ms;
}
nav.pdoc .nav-tree details[open] > summary::before { transform: rotate(90deg); }
/* a link/label inside a summary flows inline right after the marker (no block padding) */
nav.pdoc .nav-tree summary a, nav.pdoc .nav-tree summary .tree-label { display: inline; padding-left: 0; }
/* leaf rows have no marker -- pad them by the marker width so their text aligns with expandable siblings */
nav.pdoc .nav-tree li > a { padding-left: 1rem; }
/* namespace labels (Examples) are plain text, not links -- match the link color */
nav.pdoc .tree-label { color: var(--purple); }
/* indentation + rails: each nested <ul> indents once and draws a faint vertical rail spanning its
   children; hovering a node lights its subtree's rail (and its ancestors') so the subtree stands out */
nav.pdoc .nav-tree ul { padding-left: 0.6rem; border-left: 1px solid rgba(255, 255, 255, 0.18); }
nav.pdoc .nav-tree details:hover > ul { border-left-color: var(--lightblue); }
CUSTOM_CSS

cat > "$TMPL/syntax-highlighting.css" <<'SYNTAX_CSS'
pre {
  line-height: 125%;
}

td.linenos pre {
  color: #ff5555;
  background-color: #282a36;
  padding-left: 5px;
  padding-right: 5px;
}

span.linenos {
  color: #ff5555;
  background-color: #282a36;
  padding-left: 5px;
  padding-right: 5px;
}

td.linenos pre.special {
  color: #ff5555;
  background-color: #ffffc0;
  padding-left: 5px;
  padding-right: 5px;
}

span.linenos.special {
  color: #ff5555;
  background-color: #ffffc0;
  padding-left: 5px;
  padding-right: 5px;
}

/* This is for the source code docs */

.pdoc-code .hll {
  background-color: #282a36;
}

.pdoc-code {
  background: #282a36;
  color: #ff5555;
}

/* Comment */

.pdoc-code .c {
  color: #6a7aaa;
}

/* Error */
.pdoc-code .err {
  color: #ff5555;
}

/* Keyword */
.pdoc-code .k {
  color: #ff79c6;
}

/* Literal */
.pdoc-code .l {
  color: #ae81ff;
}

/* Name */
.pdoc-code .n {
  color: #f8f8f2;
}

/* Operator */
.pdoc-code {
  color: #ff79c6;
}

/* Punctuation */
.pdoc-code .p {
  color: #f8f8f2;
}

/* Comment.Hashbang */
.pdoc-code .ch {
  color: #6a7aaa;
}

/* Comment.Multiline */
.pdoc-code .cm {
  color: #6a7aaa;
}

/* Comment.Preproc */
.pdoc-code .cp {
  color: #6a7aaa;
}

/* Comment.PreprocFile */
.pdoc-code .cpf {
  color: #6a7aaa;
}

/* Comment.Single */
.pdoc-code .c1 {
  color: var(--silver);
}

/* Comment.Special */
.pdoc-code .cs {
  color: #6a7aaa;
}

/* Generic.Deleted */
.pdoc-code .gd {
  color: #6a7aaa;
}

/* Generic.Emph */
.pdoc-code .ge {
  font-style: italic;
}

/* Generic.Inserted */
.pdoc-code .gi {
  color: #a6e22e;
}

/* Genericutput */
.pdoc-code .go {
  color: whitesmoke;
}

/* Generic.Prompt */
.pdoc-code .gp {
  color: rgb(171, 138, 193);
  font-weight: bold;
}

/* Generic.Strong */
.pdoc-code .gs {
  font-weight: bold;
}

/* Generic.Subheading */
.pdoc-code .gu {
  color: #75715e;
}

/* Keyword.Constant */
.pdoc-code .kc {
  color: white;
}

/* Keyword.Declaration */
.pdoc-code .kd {
  color: rgb(171, 138, 193);
}

/* Keyword.Namespace */
.pdoc-code .kn {
  color: var(--red);
}

/* Keyword.Pseudo */
.pdoc-code .kp {
  color: rgb(171, 138, 193);
}

/* Keyword.Reserved */

.pdoc-code .kr {
  color: rgb(171, 138, 193);
}

/* Keyword.Type */

.pdoc-code .kt {
  color: #ff79c6;
}

.pdoc-code .o {
  color: var(--red);
}

/* Literal.Date */

.pdoc-code .ld {
  color: #e6db74;
}

/* Literal.Number */

.pdoc-code .m {
  color: #ae81ff;
}

/* Literal.String */

.pdoc-code .s {
  color: var(--lightblue);
}

/* Name.Attribute */

.pdoc-code .na {
  color: #a6e22e;
}

/* Name.Builtin */

.pdoc-code .nb {
  color: var(--lightblue);
}

/* Name.Class */

.pdoc-code .nc {
  color: var(--purple);
}

/* Name.Constant */

.pdoc-code .no {
  color: #ff79c6;
}

/* Name.Decorator */

.pdoc-code .nd {
  color: #8be9fd;
}

/* Name.Entity */

.pdoc-code .ni {
  color: #ff79c6;
}

/* Name.Exception */

.pdoc-code .ne {
  color: #8be9fd;
}

/* Name.Function */

.pdoc-code .nf {
  color: var(--purple);
}

/* Name.Label */

.pdoc-code .nl {
  color: #ed9d13;
}

/* Name.Namespace */

.pdoc-code .nn {
  color: #f8f8f2;
}

/* Namether */

.pdoc-code .nx {
  color: #a6e22e;
}

/* Name.Property */

.pdoc-code .py {
  color: #f8f8f2;
}

/* Name.Tag */

.pdoc-code .nt {
  color: #f92672;
}

/* Name.Variable */

.pdoc-code .nv {
  color: #f8f8f2;
}

/* Operator.Word */

.pdoc-code w {
  color: #ff79c6;
}

/* Text.Whitespace */

.pdoc-code .w {
  color: #f8f8f2;
}

/* Literal.Number.Bin */

.pdoc-code .mb {
  color: #ae81ff;
}

/* Literal.Number.Float */

.pdoc-code .mf {
  color: #ae81ff;
}

/* Literal.Number.Hex */
.pdoc-code .mh {
  color: var(--lightblue);
}

/* Literal.Number.Integer */
.pdoc-code .mi {
  color: var(--lightblue);
}

/* Literal.Numberct */
.pdoc-code .mo {
  color: #ae81ff;
}

/* Literal.String.Affix */
.pdoc-code .sa {
  color: var(--red);
}

/* Literal.String.Backtick */
.pdoc-code .sb {
  color: #e6db74;
}

/* Literal.String.Char */
.pdoc-code .sc {
  color: #e6db74;
}

/* Literal.String.Delimiter */
.pdoc-code .dl {
  color: #e6db74;
}

/* Literal.String.Doc */
.pdoc-code .sd {
  color: var(--blue);
}

/* Literal.String.Double */
.pdoc-code .s2 {
  color: var(--blue);
}

/* Literal.String.Escape */
.pdoc-code .se {
  color: var(--lightblue);
}

/* Literal.String.Heredoc */
.pdoc-code .sh {
  color: #e6db74;
}

/* Literal.String.Interpol. AKA f-strings brackets */
.pdoc-code .si {
  color: var(--lightblue);
}

/* Literal.Stringther */
.pdoc-code .sx {
  color: #e6db74;
}

/* Literal.String.Regex */
.pdoc-code .sr {
  color: var(--blue);
}

/* Literal.String.Single */
.pdoc-code .s1 {
  color: var(--blue);
}

span.linenos {
  color: rgb(59, 145, 226);
  background-color: var(--code);
}

/* Literal.String.Symbol */

.pdoc-code .ss {
  color: #e6db74;
}

/* Name.Builtin.Pseudo */
.pdoc-code .bp {
  color: whitesmoke;
}

/* Name.Function.Magic */
.pdoc-code .fm {
  color: var(--lightblue);
}

/* Name.Variable.Class */
.pdoc-code .vc {
  color: #bd93f9;
}

/* Name.Variable.Global */
.pdoc-code .vg {
  color: #f8f8f2;
}

/* Name.Variable.Instance */
.pdoc-code .vi {
  color: #ffffff;
}

/* Name.Variable.Magic */
.pdoc-code .vm {
  color: var(--lightblue);
}

/* Literal.Number.Integer.Long */
.pdoc-code .il {
  color: var(--lightblue);
}
SYNTAX_CSS

# sidebar: the hand-written prose under a "Pages" heading (friendly titles, fixed order via
# the doc_nav global), then the generated API under "Modules" as a nested package tree. The Home link
# points at guide.html (the README home page built below). The guide.* synthetic package is never
# shown verbatim; prose pages render their own markdown H1, so we drop pdoc's modulename heading +
# View-Source buttons for them, keeping those only for the real API modules.
cat > "$TMPL/module.html.jinja2" <<'JINJA'
{% extends "default/module.html.jinja2" %}
{% block module_list_link %}
    <a class="home-button" href="{{ "../" * module.modulename.count(".") }}guide.html">Home</a>
{% endblock %}
{# rename pdoc's default "Contents" (this page's own table of contents) to "Current Page" #}
{% block nav_index %}
    {% set index = module.docstring | to_markdown | to_html | attr("toc_html") %}
    {% if index %}
        <h2>Current Page</h2>
        {{ index | safe }}
    {% endif %}
{% endblock %}
{% block nav_submodules %}
    {% if doc_nav %}
    <h2>Pages</h2>
    <ul class="nav-tree">
        <li>{{ ("guide", "") | link(text="Home") }}</li>
        {% for name, title in doc_nav %}
            <li>{{ (name, "") | link(text=title) }}</li>
        {% endfor %}
        {# examples render as a collapsible "Examples" directory (numbered "1.", "2.", ... -- no
           "Example N:" prefix). Opens automatically when the current page is one of the examples. #}
        {% if examples_nav %}
        <li><details{% if module.modulename in examples_mods %} open{% endif %}><summary><span class="tree-label">Examples</span></summary>
            <ul>
                {% for name, title in examples_nav %}
                    <li>{{ (name, "") | link(text=title) }}</li>
                {% endfor %}
            </ul>
        </details></li>
        {% endif %}
    </ul>
    {% endif %}
    <h2>Modules</h2>
    {# nested package tree rendered as collapsible <details>: only the top level (microecs) shows by
       default; a node's children appear when you open it, to any depth. The path to the CURRENT page
       is emitted <details open> so you always see where you are. api_roots = top packages,
       api_children maps a module -> its submodules; `recursive` + loop(kids) walks it. #}
    <ul class="nav-tree">
    {% for name in api_roots recursive %}
        {%- set leaf = name.split(".")[-1] %}
        {%- set kids = api_children.get(name) %}
        {%- set here = module.modulename == name or module.modulename.startswith(name ~ ".") %}
        <li>
        {%- if kids %}
            <details{% if here %} open{% endif %}><summary>
                {%- if name in api_pages %}{{ (name, "") | link(text=leaf) }}{% else %}<span class="tree-label">{{ leaf }}</span>{% endif -%}
            </summary><ul>{{ loop(kids) }}</ul></details>
        {%- elif name in api_pages -%}
            {{ (name, "") | link(text=leaf) }}
        {%- else -%}
            <span class="tree-label">{{ leaf }}</span>
        {%- endif %}
        </li>
    {% endfor %}
    </ul>
{% endblock %}
{# drop pdoc's sidebar "API Documentation" member index when it would be empty: the synthetic guide.*
   prose pages have no classes/functions, and a package landing may have no own public members -- pdoc
   still emits the heading above an empty list, which reads as broken. Only show it if it has items. #}
{% block nav_members %}
    {% set _members = nav_members(module.members.values()) if module.members else "" %}
    {% if "<li" in _members %}
        <h2>API Documentation</h2>
        {{ _members }}
    {% endif %}
{% endblock %}
{% block module_info %}
    <section class="module-info">
        {% if module.modulename == "guide" or module.modulename.startswith("guide.") %}
            {{ docstring(module) }}
        {% else %}
            {{ module_name() }}
            {{ docstring(module) }}
            {{ view_source_state(module) }}
            {{ view_source_button(module) }}
            {{ view_source_code(module) }}
        {% endif %}
    </section>
{% endblock %}
JINJA

# --- render --------------------------------------------------------------------------------------
rm -rf "$OUT"
python - "$OUT" "$TMPL" <<'PY'
import sys, os, pathlib, re, tempfile, importlib, warnings
import pdoc, pdoc.render
warnings.simplefilter("ignore")

out  = pathlib.Path(sys.argv[1])
tmpl = pathlib.Path(sys.argv[2])
root = pathlib.Path(".").resolve()

# 1) API surface: every .py under microecs/ (a real package). Any module that fails to import is dropped.
def candidates(pkg):
    mods = []
    for py in sorted((root / pkg).rglob("*.py")):
        parts = py.relative_to(root).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        name = ".".join(parts)
        if name:
            mods.append(name)
    return mods

bad, api_modules = [], []
for m in candidates("microecs"):
    try:
        importlib.import_module(m)
        api_modules.append(m)
    except Exception as e:
        bad.append((m, f"{type(e).__name__}: {e}"))

# keep the `microecs` package page a clean landing: blank its __all__ in-memory (the tracked
# microecs/__init__.py is never touched) so each re-exported class is documented once, on its own
# module page (microecs.world, microecs.entity, ...) instead of ALSO dumped onto the package page.
if "microecs" in sys.modules:
    sys.modules["microecs"].__all__ = []

# 2) Prose pages: docs/source/<stem>.md, fixed sidebar order with friendly titles. Each is copied
#    into the docstring of a synthetic "guide.<stem>" module so pdoc renders it as a themed page in
#    the same sidebar as the API. README.md becomes the guide package docstring = home page.
# Two sidebar groups: flat top-level DOC_PAGES, and EXAMPLE_PAGES (rendered under a collapsible
# "Examples" directory, titled "1.", "2.", ...). Both still render as guide.<stem> pages; only their
# sidebar grouping and titles differ.
DOC_PAGES = [
    ("primitives", "Primitives"),
    ("systems",    "Systems & Per-Entity Iteration"),
    ("benchmarks", "Benchmarks"),
]
EXAMPLE_PAGES = [
    ("example-1-hello-world", "1. Hello World (raylib)"),
]
MD_PAGES = DOC_PAGES + EXAMPLE_PAGES
md_files = [(root / "docs/source" / f"{stem}.md", stem, title) for stem, title in MD_PAGES]
stem2mod = {stem: stem.replace("-", "_") for stem, _ in MD_PAGES}

# --- portable links ------------------------------------------------------------------------------
# Prose (README + docs/source/*.md) is authored with repo-relative links so it also reads on GitLab's
# file view. The GENERATED site is a separate artifact, so rewrite link TARGETS (only the `](...)`
# markdown form, never fenced code). `depth` = how many dirs deep the page sits:
#   1. self-links to our own Pages site        -> page-relative
#   2. image assets (resources/*.png|jpg|...)   -> copied into the site, depth-adjusted
#   3. inter-prose foo.md cross-links           -> the in-site guide/foo.html
#   4. external / in-site .html / pure anchors  -> left as-is
#   5. other repo paths                         -> gitlab.com blob/tree URL
# microecs' home is fixed (gitlab.com/meehai/microecs); the vendored copy's git remote points at
# robosim, so we hardcode the slug instead of reading it from git.
assets = set()  # resources images referenced by prose; copied into the site after render
_slug = "meehai/microecs"
_branch = os.environ.get("CI_DEFAULT_BRANCH") or "master"
SITE_URLS = [p.rstrip("/") + "/" for p in
            {f"https://{_slug.split('/')[0]}.gitlab.io/{_slug.split('/')[-1]}", os.environ.get("CI_PAGES_URL", "")} if p]
REPO = f"https://gitlab.com/{_slug}/-/%s/{_branch}/"      # %s -> "blob" (file) or "tree" (dir)
IMG  = re.compile(r"\.(png|jpe?g|gif|svg|webp)$", re.I)
_LINK = re.compile(r"\]\(([^)]+)\)")
def fixlinks(text, depth):
    up = "../" * depth
    def repl(m):
        tgt = m.group(1).strip()
        for pre in SITE_URLS:                                      # 1. our Pages site -> relative
            if tgt.startswith(pre):
                return f"]({up}{tgt[len(pre):] or 'index.html'})"
        if re.match(r"^(https?:|mailto:|#)", tgt):                 # 4. external / pure anchor
            return m.group(0)
        path, sep, frag = tgt.partition("#")
        clean = path[2:] if path.startswith("./") else path
        if clean.endswith(".html"):                                # 4. already in-site
            return m.group(0)
        if IMG.search(clean):                                      # 2. image asset -> copy + in-site
            assets.add(clean)
            return f"]({up}{clean}{sep}{frag})"
        stem = pathlib.PurePosixPath(clean).stem
        if clean.endswith(".md") and stem in stem2mod:             # 3. prose cross-link -> in-site html
            return f"]({up}guide/{stem2mod[stem]}.html{sep}{frag})"
        return f"]({REPO % ('tree' if clean.endswith('/') else 'blob')}{clean}{sep}{frag})"  # 5. repo url
    return _LINK.sub(repl, text)

# synthetic "guide" package: README = package docstring (home, guide.html); each prose page = a
# submodule docstring (guide.<stem>, rendered at guide/<stem>.html). The on-disk files are untouched.
# The markdown is emitted as `__doc__ = repr(text)`, NOT as a `"""..."""` literal: prose can contain
# backslashes that a raw triple-quote would mis-parse as escape sequences. repr() round-trips exactly.
guidedir = pathlib.Path(tempfile.mkdtemp())
(guidedir / "guide").mkdir()
(guidedir / "guide" / "__init__.py").write_text(
    "__doc__ = " + repr(fixlinks((root / "README.md").read_text(encoding="utf-8"), 0)) + "\n", encoding="utf-8")
guide_mods = ["guide"]
for path, stem, _title in md_files:
    (guidedir / "guide" / f"{stem2mod[stem]}.py").write_text(
        "__doc__ = " + repr(fixlinks(path.read_text(encoding="utf-8"), 1)) + "\n", encoding="utf-8")
    guide_mods.append(f"guide.{stem2mod[stem]}")
sys.path.insert(0, str(guidedir))

# 3) render everything into one themed site. doc_nav drives the flat "Pages" list;
#    examples_nav fills the collapsible "Examples" group; examples_mods lets the template auto-open
#    that group when the current page is an example.
pdoc.render.configure(template_directory=tmpl)
pdoc.render.env.globals["doc_nav"] = [(f"guide.{stem2mod[stem]}", title) for stem, title in DOC_PAGES]
pdoc.render.env.globals["examples_nav"] = [(f"guide.{stem2mod[stem]}", title) for stem, title in EXAMPLE_PAGES]
pdoc.render.env.globals["examples_mods"] = {f"guide.{stem2mod[stem]}" for stem, _title in EXAMPLE_PAGES}

# API sidebar as a package tree (not a flat list): every rendered module, plus its ancestor packages,
# wired parent -> children so the template can nest them recursively.
nodes = set(api_modules)
for m in api_modules:
    parts = m.split(".")
    for i in range(1, len(parts)):
        nodes.add(".".join(parts[:i]))
api_children = {}
for n in sorted(nodes):
    if "." in n:
        api_children.setdefault(n.rsplit(".", 1)[0], []).append(n)
pdoc.render.env.globals["api_roots"] = sorted(n for n in nodes if "." not in n)
pdoc.render.env.globals["api_children"] = api_children
pdoc.render.env.globals["api_pages"] = set(api_modules)

pdoc.pdoc(*api_modules, *guide_mods, output_directory=out)

# 4) home page = README (guide.html); make the site root redirect there, and copy every resources
#    image the prose referenced so they resolve on the site.
(out / "index.html").write_text(
    '<!doctype html><meta charset="utf-8">'
    '<meta http-equiv="refresh" content="0; url=guide.html">'
    '<link rel="canonical" href="guide.html"><title>microecs documentation</title>'
    '<a href="guide.html">Continue to the microecs documentation</a>\n', encoding="utf-8")
for rel in sorted(assets):
    src = root / rel
    if src.exists():
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())

print(f"[docs] API modules: {len(api_modules)} (dropped {len(bad)} non-importable) | "
      f"prose pages: {len(md_files)} | assets: {len(assets)}")
for m, e in bad:
    print(f"[docs]   dropped {m}: {e}")
PY

echo "[docs] built $OUT/  ->  open file://$(cd "$OUT" && pwd)/index.html"
