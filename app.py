"""Deployment entry point for the autonomous AI Movie Studio.

Run with the instance's active Python environment:
    python3 app.py
    python3 app.py --self-test

See ai_movie_studio.py's module documentation for workflow, OAuth, and host
configuration. This wrapper keeps deployment commands conventional while the
entire application remains implemented in a single standalone Python file.
"""

from ai_movie_studio import main


if __name__ == "__main__":
    main()
