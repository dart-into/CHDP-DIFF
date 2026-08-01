# CHDP-DIFF Static Demo

This repository hosts a GitHub Pages demo for:

Recognition of Newly Added Chinese Herbal Decoction Pieces via Before-and-After Image Difference.

The web page is intentionally static:

- 10 curated before-and-after image pairs are included.
- RegionSIFT + SSIM localization results were precomputed with the local Python pipeline.
- The displayed classification labels and confidence values come from the local Swin checkpoint output.
- The browser only displays the prepared images and metadata.

This choice keeps the public demo stable. GitHub Pages cannot run the Python pipeline directly, and the earlier ONNX/WASM browser model produced lower-quality demo results after conversion.

## Local Preview

From this folder:

```bash
python -m http.server 8080
```

Open:

```text
http://127.0.0.1:8080
```

Do not open `index.html` directly from the filesystem; browser security blocks JSON/image fetches from `file://`.

## GitHub Pages

The included workflow in `.github/workflows/pages.yml` deploys the static page from the repository root.
