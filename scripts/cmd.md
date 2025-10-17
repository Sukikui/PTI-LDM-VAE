# Interactive Analysis (with web server and image viewer)
```bash
python scripts/analyze_interactive.py \
    --vae-weights data/weights/autoencoder_epoch73.pth \
    --config-file config/config_train_16g_cond.json \
    --folder-edente data/edente/ \
    --folder-dente data/dente/ \
    --color-by-patient \
    --max-images 50 \
    --method tsne \
    --subtitle "Epoch 73 - Training Set"
```

# Static Analysis - UMAP (generates high-res PNG using Plotly, no server)
```bash
python scripts/analyze_static.py \
    --vae-weights data/weights/autoencoder_epoch73.pth \
    --config-file config/config_train_16g_cond.json \
    --folder-edente data/edente/ \
    --folder-dente data/dente/ \
    --output-dir results/umap_analysis \
    --color-by-patient \
    --max-images 50 \
    --method umap \
    --n-neighbors 40 \
    --min-dist 0.5 \
    --subtitle "Epoch 73 - Training Set" \
    --dpi 300
```

# Static Analysis - t-SNE (generates high-res PNG using Plotly, no server)
```bash
python scripts/analyze_static.py \
    --vae-weights data/weights/autoencoder_epoch73.pth \
    --config-file config/config_train_16g_cond.json \
    --folder-edente data/edente/ \
    --folder-dente data/dente/ \
    --output-dir results/tsne_analysis \
    --color-by-patient \
    --max-images 50 \
    --method tsne \
    --perplexity 30 \
    --subtitle "Epoch 73 - Training Set" \
    --dpi 300
```