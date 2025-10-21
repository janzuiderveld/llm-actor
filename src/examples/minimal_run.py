from __future__ import annotations

import asyncio

from app.pipeline import run_voice_pipeline


def main() -> None:
    asyncio.run(run_voice_pipeline())


if __name__ == "__main__":
    main()
