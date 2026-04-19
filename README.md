# EmbeddingPlayground

👉 **[Try the live demo](https://embedding-playground.netlify.app/)**

An interactive, browser-based tool for exploring GloVe word embeddings. Built for teaching NLP concepts.

## Quick Start

The embeddings are already pre-converted and ready to use. Just start a local server:

```bash
cd EmbeddingPlayground/
python3 -m http.server 8080
```

Then open in your browser: **http://localhost:8080**

> The app uses ES modules and must be served over HTTP; opening `index.html` directly as a `file://` URL will not work.

---

## Query Syntax

| Input | Meaning |
|-------|---------|
| `paris` | Nearest neighbors of "paris" |
| `!paris` | Farthest neighbors (least similar words) |
| `king - man + woman` | Vector arithmetic → nearest neighbors |
| `sea, ocean, lake` | Compare multiple words/expressions side by side |

---

## Visualizations

| Tab | Description |
|-----|-------------|
| **Neighbors** | Ranked nearest (or farthest) neighbors with cosine similarity bars; clickable to re-query |
| **Steps** | Step-by-step vector arithmetic: a mini bar chart per operand showing which dimensions changed most |
| **Vectors** | Raw embedding values for all query words and the result, as a bar chart |
| **Scatter (PCA)** | 2D projection of query words, result, and neighbors via power-iteration PCA |
| **Heatmap** | Full dimension grid (words × dims). blue = positive, red = negative; labels are sticky when scrolling |

### Sidebar tools

- **Similarity Calculator**: type any two words or expressions to instantly compute their cosine similarity
- **Query History**: last 20 queries, clickable to re-run; cleared with the ✕ button

---

## Models

Two models are pre-loaded (selectable via the dropdown in the header):

| Model | Dimensions | Vocabulary |
|-------|-----------|------------|
| GloVe 6B 50d | 50 | 10 000 words |
| GloVe 6B 300d | 300 | 10 000 words |

The dropdown only appears when more than one model is available.

---

## Adding a New Model

If you want to add another GloVe variant:

**1. Download the source file** from Stanford:
```
https://nlp.stanford.edu/projects/glove/
```

**2. Convert to binary format:**
```bash
python3 scripts/prepare_embeddings.py \
    --input glove.6B.100d.txt \
    --output-dir public/ \
    --name glove_100d \
    --label "GloVe 6B 100d" \
    --max-words 10000
```

This creates `public/glove_100d/embeddings.bin` + `vocab.json` and automatically updates `public/models.json`.

**3.** Restart the server. The new model will appear in the dropdown.

### `prepare_embeddings.py` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | — | Path to the GloVe `.txt` source file |
| `--output-dir` | — | Root output directory (usually `public/`) |
| `--name` | — | Subdirectory name (e.g. `glove_100d`) |
| `--label` | — | Human-readable label shown in the dropdown |
| `--max-words` | 10 000 | Vocabulary size — more = better coverage, larger file |

Recommended vocabulary sizes:

| Size | Words | File (100d) | Coverage |
|------|-------|-------------|----------|
| Small | 5 000 | ~2 MB | ~85% |
| **Default** | **10 000** | **~4 MB** | **~91%** |
| Medium | 25 000 | ~10 MB | ~96% |
| Large | 50 000 | ~20 MB | ~98% |

---

## File Structure

```
EmbeddingPlayground/
├── index.html                  # App shell
├── style.css                   # Styles
├── app.js                      # Main controller
├── lib/
│   ├── parser.js               # Expression parser
│   ├── algebra.js              # Vector math, kNN, kFarthest
│   ├── embeddings.js           # Binary loader + EmbeddingStore
│   └── pca.js                  # Power-iteration PCA (no deps)
├── vendor/
│   ├── chart.umd.min.js        # Chart.js 4 (offline copy)
│   └── chartjs-plugin-datalabels.min.js
├── public/
│   ├── models.json             # Model registry
│   ├── glove_50d/
│   │   ├── embeddings.bin      # Float32 binary embeddings
│   │   └── vocab.json          # Word list + index
│   └── glove_300d/
│       ├── embeddings.bin
│       └── vocab.json
└── scripts/
    └── prepare_embeddings.py   # Converts GloVe .txt → binary
```
