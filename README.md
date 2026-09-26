# Skin Cancer Detection

The application is split into two independently deployable folders:

| Folder | Platform | Deployment root |
| --- | --- | --- |
| `frontend/` | Vercel | Select `frontend` as the Root Directory |
| `backend/` | Render | Select `backend` as the Root Directory and use Docker |

## Deploy order

1. Deploy `backend/` to Render. Configure `ALLOWED_ORIGINS` after you know the
   Vercel URL. A trained `.pth` checkpoint is required for `/predict`; see
   `backend/README.md`.
2. Deploy `frontend/` to Vercel with `VITE_API_URL` set to the Render service
   URL (without a trailing slash).
3. Update Render's `ALLOWED_ORIGINS` to the exact Vercel production URL, then
   redeploy Render.

The model-training scripts and notebooks remain at the repository root; they
are not part of either deployment. Backend development documentation is in
`backend/DEVELOPMENT.md`.
