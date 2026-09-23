"""`python -m lending_club.testing` builds a demo bundle for CI smoke tests."""

import sys

from lending_club.testing.synthetic import build_demo_bundle

if __name__ == "__main__":
    build_demo_bundle(force="--force" in sys.argv)
