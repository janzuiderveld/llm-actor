from __future__ import annotations

import os
import time
from pathlib import Path

INBOX_PATH = Path("runtime/inbox.txt")

def append_line(line: str) -> None:
    INBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INBOX_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def main() -> None:
    append_line("P: Hello from the inbox")
    time.sleep(0.1)
    append_line("A: adding more context")


if __name__ == "__main__":
    main()
