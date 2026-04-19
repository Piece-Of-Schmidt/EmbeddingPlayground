// Power-iteration PCA for 2D projection of word vectors.
// No external dependencies. Pure JS, operates on plain arrays.
//
// Algorithm:
//   1. Center the data (subtract mean)
//   2. Find PC1 via power iteration on the covariance matrix X^T X
//   3. Deflate (project out PC1) and find PC2
//   4. Project all vectors onto [PC1, PC2]

function _dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function _norm(a) {
  return Math.sqrt(_dot(a, a));
}

function _normalize(a) {
  const n = _norm(a);
  return n > 0 ? a.map(x => x / n) : a.slice();
}

function _sub(a, b) { return a.map((x, i) => x - b[i]); }
function _scale(a, s) { return a.map(x => x * s); }
function _add(a, b) { return a.map((x, i) => x + b[i]); }

/**
 * Multiply the covariance matrix X^T X by vector v.
 * Equivalent to: sum_i  (x_i · v) * x_i
 */
function _covMatMul(centered, v) {
  const D = v.length;
  const result = new Array(D).fill(0);
  for (const x of centered) {
    const proj = _dot(x, v);
    for (let d = 0; d < D; d++) result[d] += proj * x[d];
  }
  return result;
}

function _powerIterate(centered, iters = 30) {
  const D = centered[0].length;
  // Reproducible start vector (no Math.random so PCA is deterministic)
  let v = new Array(D).fill(0).map((_, i) => Math.cos(i));
  v = _normalize(v);
  for (let i = 0; i < iters; i++) {
    v = _normalize(_covMatMul(centered, v));
  }
  return v;
}

/** Remove the component along direction pc from all vectors. */
function _deflate(centered, pc) {
  return centered.map(x => _sub(x, _scale(pc, _dot(x, pc))));
}

/**
 * Project an array of vectors to 2D using PCA.
 *
 * @param {Array<Float32Array|number[]>} vectors  each has length D
 * @param {string[]} [labels]  optional labels (for debugging)
 * @returns {Array<{x: number, y: number}>}
 */
export function pca2D(vectors) {
  const N = vectors.length;
  if (N === 0) return [];
  if (N === 1) return [{ x: 0, y: 0 }];

  const D = vectors[0].length;
  // Convert to plain number arrays for arithmetic
  const floatVecs = vectors.map(v => Array.from(v));

  // 1. Compute mean
  const mean = new Array(D).fill(0);
  for (const v of floatVecs) for (let d = 0; d < D; d++) mean[d] += v[d] / N;

  // 2. Center
  const centered = floatVecs.map(v => _sub(v, mean));

  // 3. PC1 via power iteration
  const pc1 = _powerIterate(centered, 30);

  // 4. Deflate and find PC2
  const deflated = _deflate(centered, pc1);
  const pc2 = _powerIterate(deflated, 30);

  // 5. Project
  return floatVecs.map(v => {
    const c = _sub(v, mean);
    return { x: _dot(c, pc1), y: _dot(c, pc2) };
  });
}
