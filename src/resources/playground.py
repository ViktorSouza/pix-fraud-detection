from pathlib import Path

from fastapi.responses import HTMLResponse

from constants import PIX_DEV_MODE

PLAYGROUND_FILE = Path(__file__).resolve().parent / "playground.html"


class PlaygroundResource:
    def on_get(self) -> HTMLResponse:
        page = PLAYGROUND_FILE.read_text(encoding="utf-8")
        page = page.replace("__PIX_DEV_MODE__", str(PIX_DEV_MODE).lower())
        return HTMLResponse(
            content=page,
        )
