from fastapi import FastAPI
from app.routes import files, search as search_routes

app = FastAPI(title="Parakeet Speaker Search")

app.include_router(files.router, prefix="/files", tags=["files"])
app.include_router(search_routes.router, tags=["search"])
