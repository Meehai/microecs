#!/usr/bin/env python3
"""docs/shared/build_docs.py -- the shared pdoc docs generator.

This is part of the copy-pastable "docs shared library": theme.css, custom.css,
syntax-highlighting.css, module.html.jinja2 and this script live next to each other and are
copied verbatim into EVERY project that builds docs (see docs/sync_docs_shared.sh). None of
them is project-specific; everything a project may want to customize lives in:

    <project>/docs/config.py      project-specific settings (every field documented in README.md)
    <project>/docs/custom.css     optional project-specific CSS overrides (appended to the base)
    <project>/docs/source/**/*.md the hand-written prose pages; DIRECTORY STRUCTURE = the
                                  "Pages" sidebar menu (a dir becomes a collapsible group)
    <project>/README.md           the site home page (rendered as guide.html)

How the sidebar menu is derived from docs/source:
    * every *.md is a page; its title is the first `# ` heading (fallback: the file name).
    * a directory becomes a collapsible group; its name is the group label. A group may also
      be a page itself by putting an `index.md` inside it.
    * sort order at each level: a leading number in the title (or group name) sorts first,
      numerically ("1. Getting started" < "2. Adding Velocity..."), then everything else is
      alphabetical by file/dir name. That means adding "docs/source/Examples/My Example.md"
      is ALL you need -- the menu picks it up on the next build.
    * files and dirs starting with `_` are ignored (drafts), a top-level README.md is ignored
      (the repo README is the home page), and docs/source may be absent entirely (README-only
      site).

Run from the repo root:  python docs/shared/build_docs.py <output-dir>
(Usually invoked through docs/build_docs.sh, which sets the output dir + installs pdoc.)
"""

import importlib
import importlib.util
import os
import pathlib
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import warnings

import pdoc
import pdoc.doc
import pdoc.render

warnings.simplefilter("ignore")


# --------------------------------------------------------------------------- project config ----
def _load_config(docs: pathlib.Path) -> dict:
    """Load docs/config.py (if present) as a plain key=value module, so each project's
    configuration stays human-editable and nothing custom leaks into this shared script."""
    cfg_path = docs / "config.py"
    if not cfg_path.exists():
        return {}
    spec = importlib.util.spec_from_file_location("_docs_project_config", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {k: v for k, v in vars(mod).items() if not k.startswith("_")}


def _parts(value) -> tuple:
    """Normalize an api_packages entry to a tuple of path parts ("microecs" -> ("microecs",),
    ("src", "robolib") stays ("src", "robolib")."""
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


# ------------------------------------------------------------------------------ API surface ----
def _api_candidates(base: pathlib.Path) -> list[str]:
    """Every importable module name under a package dir (__init__ files become their parent).
    The package dir itself is the top-level module prefix: microecs/world.py ->
    microecs.world, src/robolib/network/tcp.py -> robolib.network.tcp, and the base package's
    own __init__.py is the package module (microecs)."""
    mods = []
    for py in sorted(base.rglob("*.py")):
        parts = list(py.relative_to(base).with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        parts = [base.name, *parts]
        if parts[-1]:
            mods.append(".".join(parts))
    return mods


# ----------------------------------------------------------------------- prose/nav derivation ----
NUM_PREFIX = re.compile(r"^(\d+)\s*[.)]?\s*")


def _slug(name: str) -> str:
    """A filesystem path -> a python-importable module segment (and thus the page's URL
    directory). 'Examples & Tutorials' -> 'examples_tutorials'; '1. Hello World.md' ->
    'hello_world' (the leading number is an ordering hint, not part of the slug)."""
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    s = re.sub(r"^[_\d]+", "", s)          # never start a module part with a digit
    return s or "page"


def _fallback_title(stem: str) -> str:
    t = re.sub(r"^[\d\s._-]+", "", stem).replace("-", " ").replace("_", " ").strip()
    return t or stem


def _order_key(name: str, title: str):
    """Menu ordering: an explicit leading number (in the title/group name) sorts first and
    numerically; everything else is alphabetical by name. The number stays in the display
    title, so "1. Getting started" reads as today -- the number is only a sort key."""
    m = NUM_PREFIX.match(title or "")
    if m:
        return (0, int(m.group(1)), (title or "").lower())
    return (1, 0, name.lower())


class Page:
    """One prose page: a docs/source/*.md turned into a synthetic guide.* module."""

    def __init__(self, path: pathlib.Path, dirparts: tuple, stem: str, is_group: bool):
        self.path = path                    # the .md on disk
        self.dirparts = dirparts            # source-relative dir parts (without the stem)
        self.stem = stem                    # file stem ("index" for a group's index.md)
        self.is_group = is_group            # index.md of a group -> the group itself is a page
        self.slugdirs = tuple(_slug(x) for x in dirparts)
        self.slug = _slug(stem)
        self.mod = "guide." + ".".join((*self.slugdirs, self.slug) if not is_group else self.slugdirs)
        self.url = self.mod.replace(".", "/") + ".html"
        self.title = ""

    def load_title(self) -> str:
        text = self.path.read_text(encoding="utf-8")
        m = re.search(r"(?m)^#\s+(.+?)\s*$", text)
        self.title = (m.group(1).strip() if m else _fallback_title(self.stem))
        return text


def _scan_source(source: pathlib.Path):
    """Walk docs/source and build (pages, tree). `pages` maps a source-relative POSIX path
    (without .md) to its Page; `tree` is the doc_nav_tree the template walks: nodes are
    {"mod", "title", "children", "mods"} where mod is the guide module name (a link) or None
    for a label-only group, and mods lists every page-module in the node's subtree."""
    pages, by_mod = {}, {}

    def _register(page: Page):
        if page.mod in by_mod:
            raise SystemExit(
                f"[docs] docs/source: '{by_mod[page.mod]}' and "
                f"'docs/source/{'/'.join((*page.dirparts, page.stem + '.md'))}' both map to "
                f"the guide module '{page.mod}' -- rename one (the menu path is derived from "
                f"file names, so two names may not slugify identically)")
        by_mod[page.mod] = "docs/source/" + "/".join((*page.dirparts, page.stem + ".md"))
        pages["/".join(page.dirparts + (page.stem,))] = page

    def walk(dir_path: pathlib.Path, dirparts: tuple) -> list:
        items = []
        for p in sorted(dir_path.iterdir(), key=lambda p: p.name.lower()):
            if p.name.startswith("_"):     # draft file/dir, not part of the menu
                continue
            if p.is_dir():
                kids = walk(p, dirparts + (p.name,))
                if not kids and not (p / "index.md").is_file():
                    continue               # empty dir: not a menu group
                items.append(_group(p, dirparts, kids))
            elif p.suffix == ".md":
                if not dirparts and p.name.lower() == "readme.md":
                    continue               # the repo README is the home page, not a menu page
                items.append(_leaf(p, dirparts))
        items.sort(key=lambda it: _order_key(it["name"], it["title"]))
        return items

    def _leaf(p: pathlib.Path, dirparts: tuple):
        page = Page(p, dirparts, p.stem, is_group=False)
        _register(page)
        page.load_title()
        return {"mod": page.mod, "title": page.title, "children": [],
                "mods": [page.mod], "name": p.stem, "page": page}

    def _group(p: pathlib.Path, dirparts: tuple, kids: list):
        idx = p / "index.md"
        page = None
        if idx.is_file():
            page = Page(idx, dirparts + (p.name,), "index", is_group=True)
            _register(page)
            page.load_title()
        label = re.sub(r"^[\d\s._-]+", "", p.name) or p.name
        mods = [m for k in kids for m in k["mods"]]
        if page:
            mods = [page.mod] + mods
        return {"mod": page.mod if page else None, "title": label,
                "children": kids, "mods": mods, "name": p.name, "page": page}

    if not source.is_dir():
        return pages, []
    tree = walk(source, ())
    return pages, tree


def _flatten_pages(nodes: list) -> list:
    """The pages of a nav tree, in menu order (used to write the synthetic guide package)."""
    out = []
    for node in nodes:
        if node.get("page"):
            out.append(node["page"])
        out.extend(_flatten_pages(node.get("children", [])))
    return out


# -------------------------------------------------------------------------- portable links -----
def _make_link_rewriter(root: pathlib.Path, pages: dict, repo_slug, repo_branch,
                        vendored_packages):
    """Prose (README + docs/source/*.md) is authored with repo-relative links so it also reads
    on GitLab's file view. The GENERATED site is a separate artifact, so rewrite link TARGETS
    (only the `](...)` markdown form, never fenced code). Resolution order, per target:
      1. links to our own Pages site (gitlab.io/...)  -> page-relative
      2. images (resources/*.png|jpg|...)              -> copied into the site, page-relative
      3. *.md pages                                    -> the in-site guide page, relative to
         the CURRENT page (same-dir first, then a unique-stem match anywhere in the tree)
      4. external / in-site .html / pure anchors       -> left as-is
      5. other repo paths                              -> gitlab.com blob/tree URL
    Returns the rewriter closure; the referenced image assets are exposed via fixlinks.assets."""
    def _git(*a, default=""):
        try:
            return subprocess.run(["git", *a], cwd=root, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return default

    slug = repo_slug or re.sub(r"^.*gitlab\.com[:/]", "",
                               _git("remote", "get-url", "origin",
                                    default="git@gitlab.com:owner/project.git")).removesuffix(".git")
    # repo_branch: None = the checkout's git branch (robosim, where the remote IS its own
    # repo); anything else is the branch of the upstream repo (vendored libs are checked out
    # inside robosim, so their git HEAD is not their own branch -- they hardcode "master").
    branch = os.environ.get("CI_DEFAULT_BRANCH") or (repo_branch if repo_branch is not None
                                                     else _git("symbolic-ref", "--short", "HEAD")) or "master"
    site_urls = [u.rstrip("/") + "/" for u in {
        f"https://{slug.split('/')[0]}.gitlab.io/{slug.split('/')[-1]}",
        os.environ.get("CI_PAGES_URL", "")} if u]
    repo = f"https://gitlab.com/{slug}/-/%s/{branch}/"     # %s -> "blob" (file) or "tree" (dir)
    img = re.compile(r"\.(png|jpe?g|gif|svg|webp)$", re.I)
    link_re = re.compile(r"\]\((<[^>]*>|[^)\s]+)(?:\s+[^)]*)?\)")
    vendored_readme = re.compile(r"pkg/([^/]+)/README\.md")
    assets = set()

    def _up(page: Page | None) -> str:
        """'../' * how deep the current page sits below the site root (guide/<dir...>/<leaf>)."""
        return "" if page is None else "../" * (len(page.slugdirs) + 1)

    def _page_href(current: Page | None, target: Page, tail: str) -> str:
        """Relative href between two pages of the generated site (both under guide/; the home
        page is guide.html). posixpath.relpath handles any nesting depth."""
        src = current.url if current else "guide.html"
        return posixpath.relpath(target.url, start=posixpath.dirname(src)) + tail

    def _resolve_md(clean: str, page: Page | None):
        """Find the Page an *.md link points at (or None). Tries, in order: an explicit
        docs/source/... path, the current page's own directory (so '..' works too), then a
        unique stem match anywhere in the tree -- what makes 'primitives.md' links from any
        page (and from the README) keep resolving after pages move into group dirs."""
        path = clean.partition("#")[0].removesuffix(".md").strip("/")
        if not path:
            return None
        if path.startswith("docs/source/"):
            return pages.get(path[len("docs/source/"):])
        # relative to the linking page's own source dir (README = repo root)
        base = "/".join(page.dirparts) if page else ""
        cand = posixpath.normpath(posixpath.join(base or ".", path))
        if cand != ".":
            hit = pages.get(cand)
            if hit:
                return hit
        hits = [p for p in pages.values() if p.stem == path]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            print(f"[docs]   ambiguous markdown link '{clean}' ({len(hits)} pages share that "
                  f"name); left as a repo link")
        return None

    def _wrap(href: str) -> str:
        """Emit a markdown link target; angle-bracketed when it holds characters (spaces) that
        would otherwise end the destination early."""
        if re.search(r"\s", href):
            return f"](<{href}>)"
        return f"]({href})"

    def fixlinks(text: str, page: Page | None) -> str:
        def repl(m):
            tgt = (m.group(1) or "").strip()
            if tgt.startswith("<") and tgt.endswith(">"):
                tgt = tgt[1:-1].strip()
            for pre in site_urls:                                  # 1. our Pages site
                if tgt.startswith(pre):
                    rest = tgt[len(pre):]
                    return _wrap(_up(page) + (rest or "index.html"))
            if re.match(r"^(https?:|mailto:|#)", tgt):             # 4. external / pure anchor
                return m.group(0)
            path, sep, frag = tgt.partition("#")
            clean = path[2:] if path.startswith("./") else path
            if clean.endswith(".html"):                            # 4. already in-site
                return m.group(0)
            if img.search(clean):                                  # 2. image asset
                assets.add(clean)
                return _wrap(_up(page) + clean + sep + frag)
            hit = _resolve_md(clean, page)                         # 3. prose cross-link
            if hit is not None:
                return _wrap(_page_href(page, hit, sep + frag))
            vend = vendored_readme.fullmatch(clean)                # 3b. vendored lib README
            if vend and vend.group(1) in vendored_packages:
                return _wrap(_up(page) + f"vendored/{vend.group(1)}/index.html" + sep + frag)
            clean_q = urllib.parse.quote(clean, safe="/._-()&+#~")  # 5. repo url
            return _wrap(repo % ("tree" if clean.endswith("/") else "blob") + clean_q + sep + frag)
        return link_re.sub(repl, text)

    fixlinks.assets = assets
    return fixlinks


# ------------------------------------------------------------------------------------- main ----
def main() -> int:
    shared = pathlib.Path(__file__).resolve().parent      # <project>/docs/shared
    docs = shared.parent                                  # <project>/docs
    root = docs.parent                                    # repo root
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "public").resolve()
    cfg = _load_config(docs)

    # API modules (and any pre-build generators) are imported with the project's import dirs
    # ahead of site-packages; mirrored into PYTHONPATH so subprocesses (the pre-build
    # generators, the vendored builds) see the same order.
    import_paths = [root if p in ("", None) else root / p
                    for p in cfg.get("import_paths", [""])]
    for p in reversed(import_paths):
        sys.path.insert(0, str(p))
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(p) for p in import_paths] + ([existing_pythonpath] if existing_pythonpath else []))

    for cmd in cfg.get("pre_build", []):                  # project-authored regeneration steps
        subprocess.run(cmd, shell=True, cwd=root, check=True)

    bad, api_modules = [], []
    for pkg in cfg.get("api_packages", []):
        base = root.joinpath(*_parts(pkg))
        for m in _api_candidates(base):
            try:
                importlib.import_module(m)
                api_modules.append(m)
            except Exception as e:
                bad.append((m, f"{type(e).__name__}: {e}"))

    # Blank a package's re-exports in-memory only (tracked __init__.py never touched) so a
    # package that re-exports everything keeps a clean landing page instead of dumping every
    # re-exported class onto it too; each class stays documented on its own module page.
    for m in cfg.get("blank_reexports", []):
        if m in sys.modules:
            sys.modules[m].__all__ = []

    # --- prose pages + derived menu ----------------------------------------------------------
    pages, nav_tree = _scan_source(docs / "source")
    order = _flatten_pages(nav_tree)                      # pages in menu order

    fixlinks = _make_link_rewriter(root, pages, cfg.get("repo_slug"),
                                   cfg.get("repo_branch", "master"),
                                   cfg.get("vendored_packages", []))
    assets = fixlinks.assets

    # synthetic "guide" package: README = package docstring (home, guide.html); every prose
    # page = a module docstring next to it, nested exactly like docs/source. The markdown is
    # emitted as `__doc__ = repr(text)`, NOT a `"""..."""` literal: prose can contain
    # backslashes (`\x00`, `echo -e "...\n..."`) that a raw triple-quote would mis-parse.
    guidedir = pathlib.Path(tempfile.mkdtemp())
    (guidedir / "guide").mkdir()
    (guidedir / "guide" / "__init__.py").write_text(
        "__doc__ = " + repr(fixlinks((root / "README.md").read_text(encoding="utf-8"), None))
        + "\n", encoding="utf-8")
    for page in order:
        if page.is_group:                                 # group page = its package __init__.py
            d = guidedir / "guide" / pathlib.Path(*page.slugdirs)
            d.mkdir(parents=True, exist_ok=True)
            (d / "__init__.py").write_text(
                "__doc__ = " + repr(fixlinks(page.path.read_text(encoding="utf-8"), page))
                + "\n", encoding="utf-8")
        else:                                             # leaf page = a module .py file
            d = guidedir / "guide" / pathlib.Path(*page.slugdirs)
            d.mkdir(parents=True, exist_ok=True)
            if not (d / "__init__.py").exists():          # label-only group must be importable too
                (d / "__init__.py").write_text('__doc__ = ""\n', encoding="utf-8")
            (d / f"{page.slug}.py").write_text(
                "__doc__ = " + repr(fixlinks(page.path.read_text(encoding="utf-8"), page))
                + "\n", encoding="utf-8")
    sys.path.insert(0, str(guidedir))
    guide_mods = ["guide"] + [p.mod for p in order]

    # --- render -----------------------------------------------------------------------------
    # The template dir is a temp copy of the shared assets, with the project's own
    # docs/custom.css appended to the base -- the vendored library is never written to, and
    # pdoc finds theme.css/custom.css/syntax-highlighting.css where it expects them.
    tmpl = pathlib.Path(tempfile.mkdtemp())
    for f in ("theme.css", "custom.css", "syntax-highlighting.css", "module.html.jinja2"):
        shutil.copy2(shared / f, tmpl / f)
    proj_css = docs / "custom.css"
    if proj_css.exists():
        with open(tmpl / "custom.css", "a", encoding="utf-8") as fh:
            fh.write("\n/* === project overrides (docs/custom.css) === */\n"
                     + proj_css.read_text(encoding="utf-8"))

    pdoc.render.configure(template_directory=tmpl)
    pdoc.render.env.globals["doc_nav_tree"] = nav_tree
    page_titles = {"guide": "Home"}
    for p in pages.values():
        page_titles[p.mod] = p.title
    pdoc.render.env.globals["page_titles"] = page_titles

    import xml.etree.ElementTree as ET

    def promote_subtoc(toc_html):
        """markdown2 nests a page's single H1 as one top-level <li> that wraps a <ul> of its
        sub-sections. Return just that inner <ul> so the sidebar lists the sub-sections
        directly -- the page title is already shown in the 'Current Page: <title>' heading."""
        toc_html = (toc_html or "").strip()
        if not toc_html:
            return toc_html
        try:
            ul = ET.fromstring(toc_html)
        except ET.ParseError:
            return toc_html
        lis = list(ul)
        if ul.tag != "ul" or len(lis) != 1:
            return toc_html
        sub = lis[0].find("ul")
        if sub is None:
            return toc_html
        sub.set("class", "page-toc")
        return ET.tostring(sub, encoding="unicode")

    pdoc.render.env.filters["promote_subtoc"] = promote_subtoc

    # API sidebar as a package tree: every rendered module plus its ancestor packages, wired
    # parent -> children so the template can nest them recursively.
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

    # packages whose landing page renders no members (a bare `"""init file"""` __init__) are
    # shown as a plain toggle-only label in the Modules tree, not a link to an empty page.
    api_empty = set()
    for m in api_modules:
        try:
            if not pdoc.doc.Module.from_name(m).members:
                api_empty.add(m)
        except Exception:
            pass
    pdoc.render.env.globals["api_empty"] = api_empty
    pdoc.render.env.globals["vendored_pkgs"] = cfg.get("vendored_packages", [])

    # Render exactly the wanted modules. We do NOT use pdoc.pdoc() here: it walks upstream
    # packages too (a label-only group dir would get a stray auto page). html_module/index/
    # search take all_modules explicitly, so the site contains precisely guide + each page.
    all_mods = {}
    for name in [*api_modules, *guide_mods]:
        all_mods[name] = pdoc.doc.Module.from_name(name)
    for name, module in all_mods.items():
        outfile = out / f"{module.fullname.replace('.', '/')}.html"
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.write_bytes(pdoc.render.html_module(module, all_mods).encode())
    index = pdoc.render.html_index(all_mods)
    if index:
        (out / "index.html").write_bytes(index.encode())
    search = pdoc.render.search_index(all_mods)
    if search:
        (out / "search.js").write_bytes(search.encode())

    # --- home page + assets ------------------------------------------------------------------
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0; url=guide.html">'
        '<link rel="canonical" href="guide.html">'
        f'<title>{cfg.get("site_title", "docs")}</title>'
        f'<a href="guide.html">Continue to the {cfg.get("site_title", "docs")}</a>\n',
        encoding="utf-8")
    for rel in sorted(assets):
        src = root / rel
        if src.exists() and src.resolve().is_relative_to(root.resolve()):
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())

    print(f"[docs] API modules: {len(api_modules)} (dropped {len(bad)} non-importable) | "
          f"prose pages: {len(pages)} | assets: {len(assets)}")
    for m, e in bad:
        print(f"[docs]   dropped {m}: {e}")

    # --- vendored packages -------------------------------------------------------------------
    # Each vendored lib (pkg/<name>/) is documented by ITS OWN docs/build_docs.sh (same shared
    # builder, its own config.py) into out/vendored/<name>/ -- a complete, self-contained site.
    for name in cfg.get("vendored_packages", []):
        script = root / "pkg" / name / "docs" / "build_docs.sh"
        if not script.exists():
            print(f"[docs]   vendored package {name}: pkg/{name}/docs/build_docs.sh missing, skipped")
            continue
        print(f"[docs] --- vendored package: {name} ---")
        subprocess.run(["bash", str(script), str(out / "vendored" / name)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
