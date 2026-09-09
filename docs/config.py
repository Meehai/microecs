# docs/config.py -- microecs-specific settings for the shared docs builder (docs/shared/).
# Every field is documented in docs/shared/README.md; only what differs from the defaults
# is set here. The generic builder + theme live in docs/shared/ (copied from robosim's
# docs/shared/ by docs/sync_docs_shared.sh).

# ONE API surface: microecs/ (the ECS library, a real package). Any module that fails to
# import is dropped (printed at the end), so a missing optional dep never breaks the build.
api_packages = ["microecs"]

# microecs/ is imported from the repo root.
import_paths = [""]

# Blank the package's re-exports in-memory (the tracked __init__.py is never touched) so each
# re-exported class is documented once, on its own module page (microecs.world,
# microecs.entity, ...) instead of ALSO dumped onto the package page.
blank_reexports = ["microecs"]

# microecs's home is fixed (gitlab.com/meehai/microecs); the vendored copy's git remote points
# at robosim, so the slug is hardcoded instead of detected -- this site never links there.
repo_slug = "meehai/microecs"

site_title = "microecs documentation"
