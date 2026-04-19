// Vector math: cosine similarity, kNN, farthest neighbors, query computation.
// All vectors in the EmbeddingStore are pre-normalized (L2 = 1),
// so cosine similarity = dot product (no sqrt needed per query).

/** L2 norm of a Float32Array (or plain array) */
export function l2norm(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}

/** Dot product of two equal-length arrays */
export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * Compute the query result vector from a list of terms.
 * Also returns per-step intermediate vectors for the Steps visualization.
 *
 * @param {Array<{word: string, sign: number}>} terms
 * @param {import('./embeddings.js').EmbeddingStore} store
 * @returns {{
 *   resultVec: Float32Array,
 *   normalizedVec: Float32Array,
 *   steps: Array<{label, word, sign, wordVec, runningVec, changedDims}>,
 *   unknownWords: string[]
 * }}
 */
export function computeQuery(terms, store) {
  const dims = store.dims;
  const running = new Float32Array(dims);
  const steps = [];
  const unknownWords = [];

  for (let i = 0; i < terms.length; i++) {
    const { word, sign } = terms[i];
    const idx = store.getWordIndex(word);

    if (idx === -1) {
      unknownWords.push(word);
      continue;
    }

    const wordVec = store.getVector(idx); // view into normalized store
    const prevRunning = Float32Array.from(running);

    for (let d = 0; d < dims; d++) {
      running[d] += sign * wordVec[d];
    }

    // Find top-5 dimensions that changed most vs. previous running sum
    const changedDims = [];
    if (steps.length > 0) {
      const diffs = [];
      for (let d = 0; d < dims; d++) {
        diffs.push({ dim: d, delta: Math.abs(running[d] - prevRunning[d]) });
      }
      diffs.sort((a, b) => b.delta - a.delta);
      for (let k = 0; k < 5; k++) changedDims.push(diffs[k].dim);
    }

    // Build label: first term has no sign prefix, subsequent terms show +/-
    const prefix = steps.length === 0 ? '' : (sign === -1 ? '− ' : '+ ');
    steps.push({
      label:      prefix + word,
      word,
      sign,
      wordVec:    Float32Array.from(wordVec),
      runningVec: Float32Array.from(running),
      changedDims,
    });
  }

  // Normalize the result vector for cosine search
  const norm = l2norm(running);
  const normalizedVec = new Float32Array(dims);
  if (norm > 0) {
    for (let d = 0; d < dims; d++) normalizedVec[d] = running[d] / norm;
  }

  return {
    resultVec: Float32Array.from(running),
    normalizedVec,
    steps,
    unknownWords,
  };
}

/**
 * Find the k nearest neighbors of queryVec in the embedding store.
 * queryVec must be L2-normalized.
 * excludeWords: words to skip (e.g. the query words themselves).
 *
 * @returns {Array<{word: string, score: number}>} sorted descending by score
 */
export function knn(queryVec, store, k = 12, excludeWords = []) {
  return _topK(queryVec, store, k, excludeWords, false);
}

/**
 * Find the k farthest neighbors (lowest cosine similarity).
 * @returns {Array<{word: string, score: number}>} sorted ascending by score
 */
export function kfarthest(queryVec, store, k = 12, excludeWords = []) {
  return _topK(queryVec, store, k, excludeWords, true);
}

function _topK(queryVec, store, k, excludeWords, farthest) {
  const exclude = new Set(excludeWords.map(w => w.toLowerCase()));
  const dims = store.dims;

  // We maintain a fixed-size array of the best k candidates.
  // For NEAREST:  keep the k highest scores → evict the lowest  (min at index 0 after sort)
  // For FARTHEST: keep the k lowest  scores → evict the highest (max at index 0 after sort)
  const heap = [];
  const isBetter = farthest
    ? (score, heapMin) => score < heapMin   // farthest: lower is better
    : (score, heapMin) => score > heapMin;  // nearest:  higher is better
  const heapFront = farthest
    ? (a, b) => b.score - a.score  // max at [0]
    : (a, b) => a.score - b.score; // min at [0]

  for (let i = 0; i < store.size; i++) {
    const word = store.getWord(i);
    if (exclude.has(word)) continue;

    const vec = store.getVector(i);
    let score = 0;
    for (let d = 0; d < dims; d++) score += queryVec[d] * vec[d];

    if (heap.length < k) {
      heap.push({ word, score });
      if (heap.length === k) heap.sort(heapFront);
    } else if (isBetter(score, heap[0].score)) {
      heap[0] = { word, score };
      heap.sort(heapFront);
    }
  }

  // Return sorted: nearest → descending, farthest → ascending
  return farthest
    ? heap.sort((a, b) => a.score - b.score)
    : heap.sort((a, b) => b.score - a.score);
}
