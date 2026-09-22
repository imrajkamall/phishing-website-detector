# Phishing Website Detection — Vercel Version

## What this version does

`URL → safe webpage fetch → URL + webpage feature extraction → XGBoost → decision threshold → PHISHING / LEGITIMATE`

The public UI is static HTML/CSS/JavaScript. The ML inference runs in a Vercel Python Function at `/api/scan`.

## Local test

Python 3.12+ is recommended for Vercel compatibility.

```bash
pip install -r requirements.txt
uvicorn api.index:app --reload
```

Open `index.html` through a local static server or use Vercel CLI for the full routing setup.

## Vercel deployment

1. Create a GitHub repository and upload the contents of this folder.
2. Import the repository into Vercel.
3. Keep the project root at the folder containing `index.html`, `api/`, `vercel.json`, and `requirements.txt`.
4. Deploy.

The API endpoint is `/api/scan`.

## Important model note

The model was trained on the supplied 30-feature phishing dataset using the project's XGBoost configuration and a cross-validation-derived decision threshold. Some benchmark features such as domain age, web traffic, PageRank, Google Index, and inbound-link counts cannot be reliably measured by a simple page fetch, so the deployment uses neutral values for those fields and does not pretend they were measured.

This is a research/demo detector, not a guarantee of website safety.
