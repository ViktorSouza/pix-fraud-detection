from pathlib import Path

from fastapi.responses import FileResponse


PLAYGROUND_FILE = Path(__file__).resolve().parent / "playground.html"


class PlaygroundResource:
    def on_get(self) -> FileResponse:
        return FileResponse(
            path=PLAYGROUND_FILE,
            media_type="text/html",
        )
