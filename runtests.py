"""Run pytest against the local build tree instead of the editable install.

The workspace venv carries a scikit-build-core editable install of muGrid whose
ScikitBuildRedirectingFinder sits on sys.meta_path and therefore wins over
PYTHONPATH. Dropping it lets the build tree on PYTHONPATH resolve normally,
without touching site-packages.
"""
import sys

sys.meta_path = [f for f in sys.meta_path
                 if type(f).__name__ != "ScikitBuildRedirectingFinder"]

import pytest  # noqa: E402

sys.exit(pytest.main(sys.argv[1:]))
