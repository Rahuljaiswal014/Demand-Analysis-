"""Read-only SQL over the analyst database, for the Claude Code backend.

    python -m demand.sql "SELECT region, SUM(quantity) FROM sales GROUP BY 1"

Claude Code (running on the user's subscription login) is allowed to run exactly this command and nothing else;
it is the same guarded query function the API backend exposes as a tool.
"""
import sys

import duckdb

from . import ai


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print('usage: python -m demand.sql "<SELECT ...>"')
        raise SystemExit(2)
    path = ai.ensure_database()
    con = duckdb.connect(str(path), read_only=True)
    try:
        con.execute("SET enable_external_access = false")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(ai.run_query(con, " ".join(sys.argv[1:])))
    finally:
        con.close()


if __name__ == "__main__":
    main()
