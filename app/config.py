import os
from pathlib import Path
from dotenv import load_dotenv

# Always load repository-root .env, regardless of launch directory.
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT_DIR / ".env")

# Prefer explicit API_URL; otherwise build from host/port env vars.
_raw_api_url = os.getenv("API_URL")
if _raw_api_url:
    API_URL = _raw_api_url.rstrip("/")
else:
    api_host = os.getenv("API_HOST", "127.0.0.1")
    api_port = os.getenv("API_PORT", "8000")
    # 0.0.0.0 is valid for binding servers, not for browser/client requests.
    if api_host == "0.0.0.0":
        api_host = "127.0.0.1"
    API_URL = f"http://{api_host}:{api_port}"

API_KEY          = os.getenv("API_KEY", "your-secret-key")
APP_TITLE        = "JD Bias Detector"
APP_ICON         = "🔍"
CATEGORIES       = ["GENDER_CODED", "AGEIST", "EXCLUSIONARY", "ABILITY_CODED"]
CATEGORY_COLORS  = {
    "GENDER_CODED":  "#AFA9EC",
    "AGEIST":        "#F0997B",
    "EXCLUSIONARY":  "#5DCAA5",
    "ABILITY_CODED": "#FAC775",
}
CATEGORY_LABELS  = {
    "GENDER_CODED":  "Gender coded",
    "AGEIST":        "Ageist",
    "EXCLUSIONARY":  "Exclusionary",
    "ABILITY_CODED": "Ability coded",
}
