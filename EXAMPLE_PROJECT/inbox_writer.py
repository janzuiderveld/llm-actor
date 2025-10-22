from __future__ import annotations

from projects.utils import append_inbox_line


def main() -> None:
    print("Press Enter whenever a new visitor approaches the door. Ctrl+C to exit.")
    counter = 1
    try:
        while True:
            input(f"[Visitor {counter}] > ")
            append_inbox_line("P: [A new visitor steps before you.]")
            counter += 1
    except KeyboardInterrupt:
        print("\nVisitor simulator stopped.")


if __name__ == "__main__":
    main()
