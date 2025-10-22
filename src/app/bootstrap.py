from __future__ import annotations

from app.config import ConfigManager


def main() -> None:
    manager = ConfigManager()
    # Persist the freshly created (or loaded) configuration so users can edit it immediately.
    manager.save()
    print(f"Runtime configuration available at {manager.path}")


if __name__ == "__main__":
    main()
