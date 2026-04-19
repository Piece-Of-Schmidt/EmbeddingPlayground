#!/usr/bin/env python3
"""
Converts a GloVe text file (Stanford format) into the binary format
expected by GloVe Explorer.

Output (with --name mymodel):
  public/mymodel/embeddings.bin   Flat float32 binary, row-major, little-endian
  public/mymodel/vocab.json       { dims, size, words: [...] }
  public/models.json              Updated automatically with all available models

Output (without --name, legacy):
  public/embeddings.bin
  public/vocab.json

Usage:
  # First model
  python3 scripts/prepare_embeddings.py \\
      --input glove.6B.50d.txt \\
      --output-dir public/ \\
      --name glove-50d \\
      --label "GloVe 6B 50d"

  # Second model (won't overwrite the first)
  python3 scripts/prepare_embeddings.py \\
      --input glove.6B.300d.txt \\
      --output-dir public/ \\
      --name glove-300d \\
      --label "GloVe 6B 300d" \\
      --max-words 25000

Download GloVe files:
  https://nlp.stanford.edu/projects/glove/
  → glove.6B.zip  (contains 50d / 100d / 200d / 300d)
"""

import argparse
import json
import os
import struct
import sys
import base64

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input',      required=True,
                   help='Path to the GloVe .txt file (e.g. glove.6B.100d.txt)')
    p.add_argument('--output-dir', required=True,
                   help='Base output directory (usually public/)')
    p.add_argument('--name',       default=None,
                   help='Model identifier, e.g. "glove-50d". Output goes to <output-dir>/<name>/. '
                        'If omitted, files go directly into <output-dir> (legacy).')
    p.add_argument('--label',      default=None,
                   help='Human-readable name shown in the app dropdown, e.g. "GloVe 6B 50d". '
                        'Defaults to --name if not set.')
    p.add_argument('--max-words',  type=int, default=10_000,
                   help='Maximum vocabulary size (default: 10000)')
    p.add_argument('--fallback',   action='store_true',
                   help='Also write embeddings_fallback.js (for file:// use without a server)')
    return p.parse_args()


def read_glove(path, max_words):
    """
    Liest eine GloVe-Textdatei und gibt (words, matrix) zurück.
    GloVe-Dateien sind bereits nach Frequenz sortiert (häufigstes zuerst).

    Format jeder Zeile: <wort> <float1> <float2> … <floatN>
    """
    words = []
    vectors = []
    dims = None

    print(f"Lese {path} …")
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if len(words) >= max_words:
                break

            parts = line.rstrip('\n').split(' ')
            if len(parts) < 2:
                continue

            word = parts[0]
            try:
                floats = [float(x) for x in parts[1:]]
            except ValueError:
                print(f"  Zeile {i+1} übersprungen (parse-Fehler): {line[:60]!r}")
                continue

            if dims is None:
                dims = len(floats)
                print(f"  Dimensionen erkannt: {dims}")
            elif len(floats) != dims:
                print(f"  Zeile {i+1} übersprungen (falsche Dim: {len(floats)} statt {dims})")
                continue

            words.append(word)
            vectors.append(floats)

            if (i + 1) % 10_000 == 0:
                print(f"  {i+1} Zeilen gelesen …", end='\r', flush=True)

    print(f"\n  {len(words)} Wörter geladen, {dims} Dimensionen.")
    return words, vectors, dims


def write_binary(vectors, dims, path):
    """Schreibt alle Vektoren als flat little-endian float32 binary."""
    n = len(vectors)
    buf = bytearray(n * dims * 4)
    offset = 0
    for vec in vectors:
        struct.pack_into(f'{dims}f', buf, offset, *vec)
        offset += dims * 4
    with open(path, 'wb') as f:
        f.write(buf)
    size_mb = len(buf) / 1_048_576
    print(f"  {path}  ({n} × {dims} floats, {size_mb:.1f} MB)")


def write_vocab(words, dims, path):
    vocab = { 'dims': dims, 'size': len(words), 'words': words }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, separators=(',', ':'))
    size_kb = os.path.getsize(path) / 1024
    print(f"  {path}  ({len(words)} Wörter, {size_kb:.0f} KB)")


def write_fallback_js(words, vectors, dims, path):
    """
    Erzeugt eine JS-Datei, die window.GLOVE_FALLBACK setzt.
    Ermöglicht Nutzung ohne HTTP-Server (file://-Protokoll).
    Die Daten werden als Base64-kodierter float32-Binary-String eingebettet.
    """
    n = len(vectors)
    buf = bytearray(n * dims * 4)
    offset = 0
    for vec in vectors:
        struct.pack_into(f'{dims}f', buf, offset, *vec)
        offset += dims * 4

    b64 = base64.b64encode(bytes(buf)).decode('ascii')
    words_json = json.dumps(words, ensure_ascii=False, separators=(',', ':'))

    js = (
        f'// Automatisch generiert von prepare_embeddings.py\n'
        f'// {n} Wörter × {dims} Dimensionen\n'
        f'window.GLOVE_FALLBACK = {{\n'
        f'  dims: {dims},\n'
        f'  words: {words_json},\n'
        f'  dataBase64: "{b64}"\n'
        f'}};\n'
    )
    with open(path, 'w', encoding='utf-8') as f:
        f.write(js)
    size_mb = os.path.getsize(path) / 1_048_576
    print(f"  {path}  ({size_mb:.1f} MB)")


def update_models_json(base_dir, model_id, label):
    """
    Add or update an entry in public/models.json.
    Creates the file if it doesn't exist yet.
    """
    models_path = os.path.join(base_dir, 'models.json')
    models = []
    if os.path.isfile(models_path):
        with open(models_path, 'r', encoding='utf-8') as f:
            try:
                models = json.load(f)
            except json.JSONDecodeError:
                models = []

    # Update existing entry or append
    for m in models:
        if m['id'] == model_id:
            m['label'] = label
            break
    else:
        models.append({'id': model_id, 'label': label})

    with open(models_path, 'w', encoding='utf-8') as f:
        json.dump(models, f, ensure_ascii=False, indent=2)
    print(f"  {models_path}  ({len(models)} model(s) registered)")
    for m in models:
        marker = ' ← this run' if m['id'] == model_id else ''
        print(f"    · {m['id']:20s}  {m['label']}{marker}")


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"Error: input file not found: {args.input}")

    # Determine output directory
    if args.name:
        out_dir = os.path.join(args.output_dir, args.name)
    else:
        out_dir = args.output_dir

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)  # ensure base exists too

    words, vectors, dims = read_glove(args.input, args.max_words)

    if not words:
        sys.exit("Error: no words read. Is the input file valid?")

    print("\nWriting output files:")

    bin_path   = os.path.join(out_dir, 'embeddings.bin')
    vocab_path = os.path.join(out_dir, 'vocab.json')

    write_binary(vectors, dims, bin_path)
    write_vocab(words, dims, vocab_path)

    if args.fallback:
        fallback_path = os.path.join(out_dir, 'embeddings_fallback.js')
        write_fallback_js(words, vectors, dims, fallback_path)
        print(f"\nFor file:// use, add this line to index.html (before app.js):")
        src = f"public/{args.name}/embeddings_fallback.js" if args.name else "public/embeddings_fallback.js"
        print(f'  <script src="{src}"></script>')

    # Update models.json (only when --name is given)
    if args.name:
        label = args.label or args.name
        print()
        update_models_json(args.output_dir, args.name, label)

    print(f"\nDone! Now start:")
    print(f"  python3 -m http.server 8080")
    print(f"and open http://localhost:8080")


if __name__ == '__main__':
    main()
