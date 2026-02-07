# Deployment

Instructions for building and deploying the standalone article.

## Build Standalone HTML

The article at `results/article/article.html` references local PNG images via relative paths. The standalone builder inlines all local images as base64 data URIs, producing a single self-contained HTML file that can be hosted anywhere without additional assets.

```bash
uv run python -m scaling_law_analysis.article.standalone
```

This reads `results/article/article.html` and writes `results/article/article_standalone.html`.

## Deploy to GCS

### Prerequisites

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud` / `gsutil`)
- Authenticated: `gcloud auth login`
- A GCS bucket (see "Bucket Setup" below)

### Bucket Setup

<!-- TODO: fill in actual bucket name and project -->

```bash
# Create bucket (one-time)
gsutil mb -p <PROJECT_ID> -l <REGION> gs://<BUCKET_NAME>

# Enable public read access
gsutil iam ch allUsers:objectViewer gs://<BUCKET_NAME>
```

### Upload

```bash
gsutil cp \
  -h "Content-Type:text/html; charset=utf-8" \
  -h "Cache-Control:public, max-age=3600" \
  results/article/article_standalone.html \
  gs://<BUCKET_NAME>/article.html
```

The article will be available at:

```
https://storage.googleapis.com/<BUCKET_NAME>/article.html
```

### Update Workflow

1. Regenerate figures if source data changed: `uv run python -m scaling_law_analysis.article.figures`
2. Edit `results/article/article.html` as needed
3. Rebuild standalone: `uv run python -m scaling_law_analysis.article.standalone`
4. Upload: `gsutil cp -h "Content-Type:text/html; charset=utf-8" results/article/article_standalone.html gs://<BUCKET_NAME>/article.html`
