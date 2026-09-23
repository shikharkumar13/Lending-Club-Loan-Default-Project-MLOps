"""Entry point: `python -m lending_club.serving` builds the deployment bundle.

Kept separate from bundle.py so the pickled classes carry their real module
path rather than `__main__` (D-058).
"""

from lending_club.serving.bundle import package

if __name__ == "__main__":
    package()
