# Backend deployment (Render)

This folder is the Render deploy root. Create a **Web Service**, select this
directory as the service root, and choose **Docker**. Render uses the included
`Dockerfile`; the endpoints are `GET /health` and `POST /predict`.

In Render, set `ALLOWED_ORIGINS` to the exact Vercel URL, for example
`https://my-app.vercel.app`. Use `.env.production.example` as the
production-variable reference.

## Required model file

No `.pth` model checkpoint is currently in this repository. The API starts and
passes its health check without one, but prediction requests return HTTP 503.
Before deployment, put the trained checkpoint in `models/` and make it
available to Render (for example through Git LFS or a secure build-time
download). Its filename must match `MODEL_CHECKPOINT`.
