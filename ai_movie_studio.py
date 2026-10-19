"""Compatibility entry point; the complete standalone application is in app.py.

Both commands remain supported:
    python3 app.py
    python3 ai_movie_studio.py
"""

from app import *  # noqa: F401,F403 -- retain existing import/call compatibility.


if __name__ == "__main__":
    main()
