// Binary embedding loader and in-memory Float32Array store.
//
// File format (embeddings.bin):
//   Flat little-endian IEEE 754 float32 values, row-major.
//   Word i occupies bytes [i * dims * 4, (i+1) * dims * 4).
//   All vectors are L2-normalized at load time.
//
// File format (vocab.json):
//   { "dims": number, "size": number, "words": string[] }

export class EmbeddingStore {
  /**
   * @param {{ dims: number, size: number, words: string[] }} vocab
   * @param {ArrayBuffer} buffer  raw float32 binary data
   */
  constructor(vocab, buffer) {
    this.dims  = vocab.dims;
    this.size  = vocab.size;
    this.words = vocab.words;
    this._index = new Map(vocab.words.map((w, i) => [w.toLowerCase(), i]));
    this._data  = new Float32Array(buffer);
    this._normalizeAll();
  }

  /** L2-normalize every vector in place (once, at construction). */
  _normalizeAll() {
    const { dims, size } = this;
    const data = this._data;
    for (let i = 0; i < size; i++) {
      const off = i * dims;
      let norm = 0;
      for (let d = 0; d < dims; d++) norm += data[off + d] ** 2;
      norm = Math.sqrt(norm);
      if (norm > 0) for (let d = 0; d < dims; d++) data[off + d] /= norm;
    }
  }

  /**
   * Returns a Float32Array *view* into the internal buffer for word at `index`.
   * Do not mutate the returned array. Copy with Float32Array.from() if needed.
   */
  getVector(index) {
    const off = index * this.dims;
    return this._data.subarray(off, off + this.dims);
  }

  /** O(1) Map lookup. Returns -1 if unknown. */
  getWordIndex(word) {
    return this._index.get(word.toLowerCase()) ?? -1;
  }

  getWord(index) {
    return this.words[index];
  }

  hasWord(word) {
    return this._index.has(word.toLowerCase());
  }
}

/**
 * Fetch the list of available models from public/models.json.
 * Returns an array of { id, label } objects, or a default single-model list
 * if models.json does not exist (backwards-compatible).
 *
 * @returns {Promise<Array<{id: string, label: string}>>}
 */
export async function loadModelList() {
  try {
    const list = await fetch('public/models.json').then(r => r.ok ? r.json() : null);
    if (Array.isArray(list) && list.length > 0) return list;
  } catch (_) { /* fall through */ }
  // Legacy layout: single model at public/
  return [{ id: 'default', label: 'Default' }];
}

/**
 * Load embeddings for a given model id.
 * Files are expected at: public/<modelId>/embeddings.bin  and  public/<modelId>/vocab.json
 * For the special id "default", falls back to public/embeddings.bin (legacy layout).
 *
 * @param {string} modelId
 * @param {(fraction: number) => void} [onProgress]
 * @returns {Promise<EmbeddingStore>}
 */
export async function loadEmbeddings(modelId = 'default', onProgress) {
  const base = modelId === 'default' ? 'public' : `public/${modelId}`;

  // Fallback path: embeddings_fallback.js sets window.GLOVE_FALLBACK
  if (modelId === 'default' && window.GLOVE_FALLBACK) {
    return _loadFromFallback(window.GLOVE_FALLBACK, onProgress);
  }

  try {
    if (onProgress) onProgress(0);

    const bust  = `?t=${Date.now()}`;
    const vocab = await fetch(`${base}/vocab.json${bust}`, { cache: 'no-store' }).then(r => {
      if (!r.ok) throw new Error(`${base}/vocab.json: ${r.status}`);
      return r.json();
    });

    const buffer = await _fetchWithProgress(`${base}/embeddings.bin${bust}`, onProgress);
    if (onProgress) onProgress(1);
    const s = new EmbeddingStore(vocab, buffer);
    s.modelPath = `${base}/embeddings.bin`;
    return s;

  } catch (err) {
    if (modelId === 'default' && window.GLOVE_FALLBACK) {
      return _loadFromFallback(window.GLOVE_FALLBACK, onProgress);
    }
    throw new Error(
      'Could not load embeddings.\n\n' +
      'Start a local server:  python3 -m http.server 8080\n' +
      'then open http://localhost:8080\n\n' +
      'Details: ' + err.message
    );
  }
}

async function _fetchWithProgress(url, onProgress) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${url}: ${response.status}`);

  const contentLength = +response.headers.get('Content-Length');

  if (!contentLength || !response.body) {
    // No progress info available — just wait
    return response.arrayBuffer();
  }

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (onProgress) onProgress(received / contentLength);
  }

  // Concatenate all chunks into a single ArrayBuffer
  const combined = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { combined.set(chunk, offset); offset += chunk.length; }
  return combined.buffer;
}

function _loadFromFallback(fallback, onProgress) {
  const { words, dims, dataBase64 } = fallback;
  const binary = atob(dataBase64);
  const buffer = new ArrayBuffer(binary.length);
  const view   = new Uint8Array(buffer);
  for (let i = 0; i < binary.length; i++) view[i] = binary.charCodeAt(i);
  if (onProgress) onProgress(1);
  const vocab = { dims, size: words.length, words };
  return Promise.resolve(new EmbeddingStore(vocab, buffer));
}
