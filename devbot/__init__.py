"""DevBot - a terminal coding agent powered by the DeepSeek API."""

import sys

# On Windows the default stdout encoding is often cp1252, which can't render
# the Unicode glyphs (box-drawing, arrows, emoji) that the dashboard and
# thinking indicator use.  Reconfigure to UTF-8 when possible.
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

__version__ = "0.1.0"
