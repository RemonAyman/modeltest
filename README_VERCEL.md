Deployment to Vercel (serverless)

Notes:
- Vercel runs Python functions under `/api/*.py`. We wrap the Flask WSGI `app` using `vercel-wsgi`.
- Large ML dependencies (pandas, scikit-learn, joblib) may increase the package size; serverless limits on Vercel might be hit. If you hit size/time limits, consider Render or Railway instead.

Steps to deploy:
1. Ensure `requirements.txt` includes all packages (already added).
2. Commit and push to your GitHub remote connected to Vercel.
3. In the Vercel dashboard, import the Git repository and deploy.

If deployment fails due to package size or build timeouts:
- Option A: Move heavy ML work to a separate hosted API (Render/Railway) and keep frontend on Vercel.
- Option B: Use a pre-built Docker service (e.g., Render with Docker) that supports larger images.

Local test using `vercel dev` (requires Vercel CLI):

1. Install Vercel CLI: `npm i -g vercel`
2. From project root run:

```bash
vercel dev
```

This will run your `/api/index.py` as a serverless function locally.

Troubleshooting:
- If `model_data.pkl` or `cleaned_transport_dataset.csv` are not present in the repository, predictions will return errors; either include these files in the repo or host them in accessible storage and update the code to fetch them at runtime.
- If a package fails to install on Vercel because of build time or size, use Render or Railway instead.
