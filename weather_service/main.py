"""
Weather Service - FastAPI Application
Serves weather forecasts from MongoDB
Port: 5002
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import forecast, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print("🌤️  Weather Service starting up...")
    yield
    print("🌤️  Weather Service shutting down...")


app = FastAPI(
    title="Weather Forecast Service",
    description="REST API for serving FuXi-S2S weather forecasts",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for external access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(forecast.router, prefix="/api/v1", tags=["Forecast"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Weather Forecast Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5002))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
