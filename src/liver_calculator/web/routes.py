from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from liver_calculator.config import get_model_config
from liver_calculator.services.scoring import get_model_summary


router = APIRouter(include_in_schema=False)

web_dir = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(web_dir / "templates"))


def is_model_ready() -> bool:
    model_config = get_model_config()
    return model_config.model_path.exists() and model_config.metadata_path.exists()


@router.get("/", response_class=HTMLResponse)
def read_index(request: Request) -> HTMLResponse:
    model_summary = None
    if is_model_ready():
        try:
            model_summary = get_model_summary()
        except (FileNotFoundError, KeyError, ValueError, OSError, TypeError):
            model_summary = None

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_title": "Liver Meta Calculator",
            "model_ready": is_model_ready(),
            "model_summary": model_summary,
        },
    )
