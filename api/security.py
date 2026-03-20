"""
security.py — minimal API key authentication.
"""
import hmac
import os
from typing import Optional

from fastapi import Header, HTTPException


API_KEY = os.getenv("API_KEY", "your-secret-key")


def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing x-api-key header")
    if not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key")
