"""DevGuide main application entry point."""

from fastapi import FastAPI

app = FastAPI(
    title="DevGuide",
    description="Local developer documentation assistant",
    version="0.1.0",
)
