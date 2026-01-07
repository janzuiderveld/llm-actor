import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from projects.utils import acquire_process_lock


def test_acquire_process_lock_creates_and_releases_lock(tmp_path) -> None:
    lock_path = tmp_path / "watcher.lock"
    lock = acquire_process_lock(lock_path)
    assert lock is not None
    assert lock_path.exists()
    assert lock_path.read_text(encoding="utf-8").strip() == str(os.getpid())
    lock.release()
    assert not lock_path.exists()


def test_acquire_process_lock_returns_none_when_active(tmp_path) -> None:
    lock_path = tmp_path / "watcher.lock"
    lock = acquire_process_lock(lock_path)
    assert lock is not None
    assert acquire_process_lock(lock_path) is None
    lock.release()


def test_acquire_process_lock_reclaims_stale_lock(tmp_path) -> None:
    lock_path = tmp_path / "watcher.lock"
    lock_path.write_text("0", encoding="utf-8")
    lock = acquire_process_lock(lock_path)
    assert lock is not None
    lock.release()
