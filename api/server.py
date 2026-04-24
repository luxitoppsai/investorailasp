"""FastAPI web service + Telegram bot (polling mode).

One process handles both:
  - HTTP  → GET /health   (keep-alive ping from cron-job.org)
  - Telegram updates      (python-telegram-bot polling via asyncio)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.handlers import build_application

logger = logging.getLogger(__name__)

_ptb_app = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ptb_app
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    _ptb_app = build_application(token)

    await _ptb_app.initialize()
    await _ptb_app.start()
    await _ptb_app.updater.start_polling(drop_pending_updates=True)
    logger.info("Telegram bot polling started")

    yield

    await _ptb_app.updater.stop()
    await _ptb_app.stop()
    await _ptb_app.shutdown()
    logger.info("Telegram bot stopped")


app = FastAPI(title="AsesorFinan Bot", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}
