# GitHub Pages ONNX/WASM Demo

This folder is a static GitHub Pages version of the RegionSIFT demo.

It does not run the full Python pipeline in the browser. Instead:

- RegionSIFT + SSIM detection boxes and visualizations are precomputed for 3 curated image pairs.
- The classifier is exported to `models/densenet169.onnx`.
- The browser runs DenseNet169 classification on detected regions through ONNX Runtime Web's WASM backend.
- The ONNX model is about 52 MB, which is under GitHub's 100 MB per-file limit.

## Local Preview

From this folder:

```bash
python -m http.server 8080
```

Open:

```text
http://127.0.0.1:8080
```

Do not open `index.html` directly from the filesystem; browser security blocks model/data fetches from `file://`.

## Deploy To GitHub Pages

Create an empty GitHub repository, then run:

```bash
git init
git add .
git commit -m "Add ONNX WASM demo"
git branch -M main
git remote add origin https://github.com/<your-name>/<your-repo>.git
git push -u origin main
```

In GitHub, open the repository settings:

1. Go to **Settings > Pages**.
2. Set **Source** to **Deploy from a branch**.
3. Select **main** and **/**.
4. Save and wait for the Pages URL to appear.
