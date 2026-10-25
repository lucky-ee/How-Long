const round1 = (v) => Math.round(v * 10) / 10;
const show = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(1));
const unit = (v, one, many) => (v === 1 ? one : many);

/** Hours -> { num, unit } in the most readable unit. Infinity means the curve never reaches 50%. */
export function durationParts(h) {
  if (h == null || !Number.isFinite(h)) return { num: "2+", unit: "years" };
  if (h < 1) {
    const m = Math.max(1, Math.round(h * 60));
    return { num: String(m), unit: unit(m, "minute", "minutes") };
  }
  if (h < 36) {
    const v = h < 10 ? round1(h) : Math.round(h);
    return { num: show(v), unit: unit(v, "hour", "hours") };
  }
  const d = h / 24;
  if (d < 14) {
    const v = d < 10 ? round1(d) : Math.round(d);
    return { num: show(v), unit: "days" };
  }
  if (d < 60) {
    const v = round1(d / 7);
    return { num: show(v), unit: "weeks" };
  }
  if (d < 730) {
    const m = d / 30.44;
    const v = m < 10 ? round1(m) : Math.round(m);
    return { num: show(v), unit: unit(v, "month", "months") };
  }
  const v = round1(d / 365);
  return { num: show(v), unit: "years" };
}

export const duration = (h) => {
  const p = durationParts(h);
  return `${p.num} ${p.unit}`;
};

export function pct(p) {
  if (p < 0.005) return "<1%";
  if (p > 0.995) return ">99%";
  return `${Math.round(p * 100)}%`;
}

export const sentenceCase = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : "");

export function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

/** Local wall-clock time as "YYYY-MM-DDTHH:MM:SS" (the model's `opened` is SF local time). */
export function localISO(date) {
  const p = (n) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${p(date.getMonth() + 1)}-${p(date.getDate())}` +
    `T${p(date.getHours())}:${p(date.getMinutes())}:${p(date.getSeconds())}`;
}
