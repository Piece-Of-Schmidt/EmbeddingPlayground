// Expression parser for word embedding queries.
// Supported syntax:
//   "paris"              → nearest neighbors of "paris"
//   "!paris"             → farthest neighbors of "paris"
//   "king - man + woman" → vector arithmetic, then nearest neighbors
//   "king - man + woman = ?" → same (= ? is cosmetic)
//   "! king - man + woman"  → farthest neighbors of arithmetic result
//   "human + male + adult + unmarried" → sum, then nearest neighbors

export const MODE_NEAREST  = 'NEAREST';
export const MODE_FARTHEST = 'FARTHEST';

const T_WORD  = 'WORD';
const T_PLUS  = 'PLUS';
const T_MINUS = 'MINUS';

function tokenize(raw) {
  const tokens = [];
  // Split on whitespace, but also split tokens like "king-man" on - if no spaces
  // We do a two-pass: first split on spaces, then re-split tokens containing +/-
  const spaceParts = raw.trim().split(/\s+/);

  for (const part of spaceParts) {
    if (!part) continue;
    // Split embedded +/- (e.g. "king-man+woman" with no spaces)
    // We use a lookahead/lookbehind split that keeps the delimiters
    const subParts = part.split(/(?=[+\-])|(?<=[+\-])/);
    for (const sub of subParts) {
      if (!sub) continue;
      if (sub === '+') { tokens.push({ type: T_PLUS }); }
      else if (sub === '-') { tokens.push({ type: T_MINUS }); }
      else if (/^[a-zA-Z][a-zA-Z0-9'_\-.]*$/.test(sub)) {
        tokens.push({ type: T_WORD, value: sub.toLowerCase() });
      }
      // silently skip unknown characters (=, ?, digits, etc.)
    }
  }
  return tokens;
}

/**
 * Parse a query string into a structured query object.
 * @param {string} input
 * @returns {{ terms: Array<{word: string, sign: number}>, mode: string }}
 */
export function parse(input) {
  let raw = input.trim();

  // Strip trailing "= <anything>" (e.g. "= ?", "= xyz", "=?")
  raw = raw.replace(/\s*=\s*\S*\s*$/, '').trim();

  // Detect farthest mode: leading "!" with optional space
  let mode = MODE_NEAREST;
  if (raw.startsWith('!')) {
    mode = MODE_FARTHEST;
    raw = raw.slice(1).trim();
  }

  const tokens = tokenize(raw);
  const terms = [];
  let sign = +1;

  for (const tok of tokens) {
    if      (tok.type === T_PLUS)  { sign = +1; }
    else if (tok.type === T_MINUS) { sign = -1; }
    else if (tok.type === T_WORD)  {
      terms.push({ word: tok.value, sign });
      sign = +1; // reset after each word
    }
  }

  return { terms, mode };
}

/**
 * Returns a human-readable description of the query.
 */
export function describeQuery({ terms, mode }) {
  if (terms.length === 0) return '';
  const expr = terms.map(({ word, sign }, i) => {
    if (i === 0) return sign === -1 ? `−${word}` : word;
    return (sign === -1 ? ' − ' : ' + ') + word;
  }).join('');
  return mode === MODE_FARTHEST ? `!(${expr})` : expr;
}
