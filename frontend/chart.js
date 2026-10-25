// "Chance it's closed by time t" curve: log-time x axis, animated between predictions.

const W = 640, H = 260;
const M = { l: 44, r: 16, t: 14, b: 30 };
const IW = W - M.l - M.r, IH = H - M.t - M.b;
const TICKS = [
  [1 / 6, "10m"], [1, "1h"], [6, "6h"], [24, "1d"], [72, "3d"], [168, "1w"],
  [720, "1mo"], [2160, "3mo"], [8760, "1y"],
];
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

// Same grid the API predicts on (models.MEDIAN_GRID): 400 log-spaced points, 3 minutes to 2 years.
export const GRID = Array.from({ length: 400 }, (_, i) => 0.05 * Math.pow(17520 / 0.05, i / 399));
const LOG_G0 = Math.log(GRID[0]), LOG_SPAN = Math.log(GRID[GRID.length - 1]) - LOG_G0;

function stepAt(xs, ys, t) {
  if (t < xs[0]) return 1;
  let lo = 0, hi = xs.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (xs[mid] <= t) lo = mid; else hi = mid - 1;
  }
  return ys[lo];
}

/** Resample any survival curve onto GRID as P(closed by t). */
export const closedOnGrid = (hours, survival) => GRID.map((t) => 1 - stepAt(hours, survival, t));

export function valueAt(closed, t) {
  const f = ((Math.log(t) - LOG_G0) / LOG_SPAN) * (GRID.length - 1);
  if (f <= 0) return closed[0];
  if (f >= GRID.length - 1) return closed[GRID.length - 1];
  const i = Math.floor(f);
  return closed[i] + (closed[i + 1] - closed[i]) * (f - i);
}

export function quantile(closed, q) {
  const i = closed.findIndex((p) => p >= q);
  return i < 0 ? Infinity : GRID[i];
}

function viewFor(closed) {
  const a = quantile(closed, 0.03), b = quantile(closed, 0.97);
  const hi = Math.min(17520, Math.max(48, Number.isFinite(b) ? b * 2 : 17520));
  let lo = Math.max(0.25, (Number.isFinite(a) ? a : 1) / 3);
  if (hi / lo < 40) lo = Math.max(0.05, hi / 40);
  return { lo, hi };
}

const ease = (k) => (k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2);
const lerpLog = (a, b, e) => Math.exp(Math.log(a) + (Math.log(b) - Math.log(a)) * e);

export class CurveChart {
  constructor(root, formatHours) {
    this.fmt = formatHours;
    this.cur = null;
    this.hoverX = null;
    root.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Chance the request is closed, over time">
        <defs>
          <linearGradient id="hl-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" style="stop-color:var(--accent);stop-opacity:.34"/>
            <stop offset="100%" style="stop-color:var(--accent);stop-opacity:0"/>
          </linearGradient>
          <clipPath id="hl-clip"><rect x="${M.l}" y="${M.t - 3}" width="${IW}" height="${IH + 6}"/></clipPath>
        </defs>
        <g class="axes"></g>
        <g clip-path="url(#hl-clip)">
          <path class="area" fill="url(#hl-fill)"/>
          <path class="line"/>
        </g>
        <g class="median"><line/><circle r="4.5"/></g>
        <g class="hover"><line/><circle r="5.5"/></g>
      </svg>
      <div class="chart-tip" role="status"></div>`;
    this.svg = root.querySelector("svg");
    this.axes = root.querySelector(".axes");
    this.area = root.querySelector(".area");
    this.line = root.querySelector(".line");
    this.median = root.querySelector(".median");
    this.hover = root.querySelector(".hover");
    this.tip = root.querySelector(".chart-tip");
    this.root = root;

    this.svg.addEventListener("pointermove", (e) => {
      const r = this.svg.getBoundingClientRect();
      const sx = ((e.clientX - r.left) * W) / r.width;
      this.hoverX = Math.min(Math.max(sx, M.l), M.l + IW);
      this.drawHover();
    });
    this.svg.addEventListener("pointerleave", () => {
      this.hoverX = null;
      root.classList.remove("is-hovering");
    });
  }

  update(closed) {
    const target = { closed: Float64Array.from(closed), ...viewFor(closed) };
    cancelAnimationFrame(this.raf);
    if (!this.cur || reduceMotion) {
      this.cur = target;
      this.draw();
      return;
    }
    const from = this.cur, t0 = performance.now(), dur = 750;
    const frame = (now) => {
      const k = Math.min(1, (now - t0) / dur), e = ease(k);
      this.cur = {
        closed: from.closed.map((v, i) => v + (target.closed[i] - v) * e),
        lo: lerpLog(from.lo, target.lo, e),
        hi: lerpLog(from.hi, target.hi, e),
      };
      this.draw();
      if (k < 1) this.raf = requestAnimationFrame(frame);
    };
    this.raf = requestAnimationFrame(frame);
  }

  scales() {
    const { lo, hi } = this.cur;
    const l0 = Math.log(lo), span = Math.log(hi) - l0;
    return {
      x: (t) => M.l + ((Math.log(t) - l0) / span) * IW,
      y: (p) => M.t + (1 - p) * IH,
      t: (px) => Math.exp(l0 + ((px - M.l) / IW) * span),
    };
  }

  draw() {
    const { closed, lo, hi } = this.cur;
    const { x, y } = this.scales();

    let d = "", first = -1, last = -1;
    for (let i = 0; i < GRID.length; i++) {
      if (GRID[i] < lo / 1.3 || GRID[i] > hi * 1.3) continue;
      if (first < 0) first = i;
      last = i;
      d += `${d ? "L" : "M"}${x(GRID[i]).toFixed(1)},${y(closed[i]).toFixed(1)}`;
    }
    this.line.setAttribute("d", d);
    this.area.setAttribute("d",
      `${d}L${x(GRID[last]).toFixed(1)},${y(0)}L${x(GRID[first]).toFixed(1)},${y(0)}Z`);

    const yTicks = [0, 0.5, 1].map((p) =>
      `<line x1="${M.l}" x2="${M.l + IW}" y1="${y(p)}" y2="${y(p)}"/>` +
      `<text x="${M.l - 8}" y="${y(p) + 4}" text-anchor="end">${p * 100}%</text>`);
    const xTicks = TICKS.filter(([t]) => t > lo * 1.08 && t < hi / 1.08).map(([t, label]) => {
      const px = x(t).toFixed(1);
      return `<line class="v" x1="${px}" x2="${px}" y1="${M.t}" y2="${M.t + IH}"/>` +
        `<text x="${px}" y="${H - 8}" text-anchor="middle">${label}</text>`;
    });
    this.axes.innerHTML = yTicks.join("") + xTicks.join("");

    const tm = quantile(closed, 0.5);
    const showMedian = tm >= lo && tm <= hi;
    this.median.style.opacity = showMedian ? 1 : 0;
    if (showMedian) {
      const mx = x(tm);
      const [ml, mc] = this.median.children;
      ml.setAttribute("x1", mx); ml.setAttribute("x2", mx);
      ml.setAttribute("y1", y(0.5)); ml.setAttribute("y2", y(0));
      mc.setAttribute("cx", mx); mc.setAttribute("cy", y(0.5));
    }
    this.drawHover();
  }

  drawHover() {
    if (this.hoverX == null || !this.cur) return;
    const { y, t: toHours } = this.scales();
    const t = toHours(this.hoverX);
    const p = valueAt(this.cur.closed, t);
    const [hl, hc] = this.hover.children;
    hl.setAttribute("x1", this.hoverX); hl.setAttribute("x2", this.hoverX);
    hl.setAttribute("y1", M.t); hl.setAttribute("y2", M.t + IH);
    hc.setAttribute("cx", this.hoverX); hc.setAttribute("cy", y(p));

    const r = this.svg.getBoundingClientRect();
    const left = (this.hoverX * r.width) / W;
    this.tip.innerHTML = `<strong>${Math.round(p * 100)}%</strong> closed within ${this.fmt(t)}`;
    const half = this.tip.offsetWidth / 2 + 4;
    this.tip.style.left = `${Math.min(Math.max(left, half), r.width - half)}px`;
    this.tip.style.top = `${(y(p) * r.height) / H}px`;
    this.root.classList.add("is-hovering");
  }
}
