"""``python -m data_parser_app``: GUI by default; pass ``-f`` for CLI."""

from data_parser_app.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
