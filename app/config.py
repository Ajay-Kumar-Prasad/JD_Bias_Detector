import os
from dotenv import load_dotenv

load_dotenv()

API_URL          = os.getenv("API_URL", "http://localhost:8000")
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
