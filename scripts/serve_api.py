from pathlib import Path
import os
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    import uvicorn

    from liver_calculator.config import get_app_settings

    settings = get_app_settings()
    reload_enabled = settings.reload or os.getenv("APP_RELOAD") is None
    uvicorn.run(
        "liver_calculator.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=reload_enabled,
    )


if __name__ == "__main__":
    main()
