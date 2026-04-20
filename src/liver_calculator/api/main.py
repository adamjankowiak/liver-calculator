from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from liver_calculator.api.routes_predict import router as predict_router
from liver_calculator.config import get_app_settings
from liver_calculator.web.routes import router as web_router


settings = get_app_settings()
app = FastAPI(
    title=settings.title,
    version="0.1.0",
    description="Inference API and local web UI for the liver fibrosis meta-calculator.",
)

static_dir = Path(__file__).resolve().parents[1] / "web" / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.include_router(predict_router)
app.include_router(web_router)
