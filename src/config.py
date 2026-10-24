from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
MANIFEST = RAW_DIR / "_manifest.json"

# DataSF 311 Cases dataset (https://data.sfgov.org/d/vw6y-z8j6)
SODA_BASE = "https://data.sfgov.org"
DATASET_ID = "vw6y-z8j6"
PAGE_SIZE = 50_000

API_COLUMNS = [
    "service_request_id",
    "requested_datetime",
    "closed_date",
    "updated_datetime",
    "status_description",
    "status_notes",
    "agency_responsible",
    "service_name",
    "service_subtype",
    "service_details",
    "address",
    "supervisor_district",
    "analysis_neighborhood",
    "lat",
    "long",
    "source",
    "media_url",
]

SF_LAT = (37.60, 37.84)
SF_LON = (-122.53, -122.35)

MIN_VALID_DATE = "2008-07-01"

CATEGORY_ALIASES = {
    "encampment": "encampments",
}
SOURCE_ALIASES = {
    "mobile/open311": "mobile",
}
DROP_SOURCES = {"test"}