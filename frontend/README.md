# How Long? — frontend

Static site, no build step: plain HTML, CSS and ES modules.

## Run locally

```sh
cd frontend
python3 -m http.server 8000      # then open http://localhost:8000
```

(ES modules don't load from `file://`, so it needs any static server.)

## Deploy

**With the API (recommended):** `src/api.py` serves this folder at `/` and the model at
`/api`, and points the page at it automatically. See the top-level README.

**On its own:** upload the `frontend/` folder as-is to any static host (Netlify, Vercel,
Cloudflare Pages, GitHub Pages). No build command, publish directory `frontend`. To use a
separately hosted API, set it in `config.js`:

```js
window.HOWLONG_CONFIG = { apiBase: "https://your-api.example.com/api" };
```

With `apiBase` empty, or if the API can't be reached, the site runs in **demo mode** on the
true generating model from `src/synth.py`, and shows a badge saying so.

The API (`src/api.py`) serves, relative to `apiBase`:

| Route | Returns |
|---|---|
| `GET /options` | `options.json` written by `train.py` (`serving_options`) |
| `GET /report` | `report.json` written by `train.py` |
| `POST /predict` | a survival curve for one request (below) |

`POST /predict` body: the fields in `features.INPUT_COLUMNS`:

```json
{
  "opened": "2026-10-25T10:37:21",
  "category": "street defects",
  "request_type": "pothole",
  "request_details": null,
  "source": "mobile",
  "supervisor_district": 9,
  "has_photo": false,
  "backlog_at_filing": 42
}
```

`supervisor_district` and `backlog_at_filing` are `null` when the user picks "Not sure".
The backlog comes from `options.backlog["<request_type>|<district>"]` (a missing key means 0).

Response: `model.survival(df, MEDIAN_GRID)` for that row, plus the median read off that curve:

```json
{ "median_hours": 33.1, "hours": [0.05, "...", 17520], "survival": [1.0, "...", 0.0] }
```

`median_hours` may be `null` (the curve never reaches 50%). Any increasing `hours` grid works;
the page resamples it.
