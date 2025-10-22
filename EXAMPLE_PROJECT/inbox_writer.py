"""Small helper that simulates new visitors by writing to the shared inbox."""

from __future__ import annotations

from projects.utils import append_inbox_line


def main() -> None:
    print("Press Enter whenever a new visitor approaches the door. Ctrl+C to exit.")
    counter = 1
    try:
        while True:
            # Blocking input keeps the loop simple: we wait until you press Enter.
            input(f"[Visitor {counter}] > ")
            # Each visitor announcement is written using the shared helper.
            append_inbox_line("P: [A new visitor steps before you.]")
            counter += 1
    except KeyboardInterrupt:
        print("\nVisitor simulator stopped.")


if __name__ == "__main__":
    main()
