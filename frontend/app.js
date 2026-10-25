import { demo } from "./demo.js";
import { CurveChart, closedOnGrid, quantile, valueAt } from "./chart.js";
import { duration, durationParts, esc, localISO, pct, sentenceCase } from "./format.js";

const $ = (sel) => document.querySelector(sel);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

// ---------------------------------------------------------------- backend

const API = (window.HOWLONG_CONFIG?.apiBase || "").replace(/\/+$/, "");

async function getJSON(path, init) {
  const res = await fetch(API + path, init);
  if (!res.ok) throw new Error(`${path} returned HTTP ${res.status}`);
  return res.json();
}

const api = {
  options: () => getJSON("/options"),
  report: () => getJSON("/report"),
  predict: (body, signal) => getJSON("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  }),
};

let backend = API ? api : demo;

// ---------------------------------------------------------------- state

const state = {
  options: null,
  type: null,      // { request_type, category, cases }
  district: "",    // "" = not sure
  source: null,
  photo: false,
  when: null,      // null = now, else "YYYY-MM-DDTHH:MM:SS"
};

const SOURCE_LABELS = { mobile: "App", phone: "Phone", web: "Web" };
const MODEL_LABELS = {
  linear: "Penalized Cox",
  xgboost: "XGBoost AFT",
  km_baseline: "Kaplan–Meier",
  synthetic: "Synthetic truth",
};
const MILESTONES = [[24, "1 day"], [72, "3 days"], [168, "1 week"], [720, "1 month"]];

// ---------------------------------------------------------------- boot

async function boot() {
  setupTheme();
  let options;
  try {
    options = await backend.options();
  } catch (err) {
    console.warn("API unavailable, falling back to demo data:", err);
    backend = demo;
    options = await demo.options();
    setBadge("API offline · demo data");
  }
  if (backend === demo && $("#mode-badge").hidden) setBadge("Demo · synthetic data");

  state.options = options;
  restoreFromURL();
  setupTypePicker();
  setupDistrict();
  setupSource();
  setupPhoto();
  setupWhen();
  setupShare();

  chart = new CurveChart($("#chart"), duration);
  document.body.classList.add("is-ready");
  predict();

  backend.report().then(renderInsights).catch((err) => {
    console.warn("No report available:", err);
    $("#insights").hidden = true;
  });
  setupReveal();
}

function setBadge(text) {
  const badge = $("#mode-badge");
  badge.textContent = text;
  badge.hidden = false;
}

// ---------------------------------------------------------------- URL state

function restoreFromURL() {
  const q = new URLSearchParams(location.search);
  const types = state.options.request_types;
  state.type = types.find((t) => t.request_type === q.get("type")) ??
    [...types].sort((a, b) => b.cases - a.cases)[0];
  const d = q.get("district");
  state.district = state.options.districts.some((x) => String(x.district) === d) ? d : "";
  const sources = state.options.sources;
  state.source = sources.includes(q.get("source")) ? q.get("source")
    : sources.includes("mobile") ? "mobile" : sources[0];
  state.photo = q.get("photo") === "1";
  state.when = /^\d{4}-\d\d-\d\dT\d\d:\d\d/.test(q.get("when") ?? "") ? q.get("when") : null;
}

function syncURL() {
  const q = new URLSearchParams();
  q.set("type", state.type.request_type);
  if (state.district) q.set("district", state.district);
  q.set("source", state.source);
  if (state.photo) q.set("photo", "1");
  if (state.when) q.set("when", state.when);
  history.replaceState(null, "", `${location.pathname}?${q}`);
}

function changed() {
  syncURL();
  schedulePredict();
}

// ---------------------------------------------------------------- form controls

function setupTypePicker() {
  const input = $("#type-input"), list = $("#type-list"), quick = $("#quick-picks");
  const items = [...state.options.request_types].sort((a, b) => b.cases - a.cases);
  let matches = [], active = -1;

  const highlight = (text, q) => {
    const i = q ? text.toLowerCase().indexOf(q) : -1;
    if (i < 0) return esc(text);
    return esc(text.slice(0, i)) + `<mark>${esc(text.slice(i, i + q.length))}</mark>` +
      esc(text.slice(i + q.length));
  };

  const render = (q) => {
    list.innerHTML = matches.length
      ? matches.map((it, i) => `
          <li role="option" id="type-opt-${i}" data-i="${i}" aria-selected="${i === active}">
            <span>${highlight(sentenceCase(it.request_type), q)}</span>
            <small>${highlight(sentenceCase(it.category), q)}</small>
          </li>`).join("")
      : `<li class="empty">No matching request types</li>`;
    input.setAttribute("aria-activedescendant", active >= 0 ? `type-opt-${active}` : "");
    list.querySelector('[aria-selected="true"]')?.scrollIntoView({ block: "nearest" });
  };

  const open = () => {
    const q = input.value.trim().toLowerCase();
    matches = items.filter((it) => !q || it.request_type.includes(q) || it.category.includes(q)).slice(0, 8);
    active = matches.length ? 0 : -1;
    list.hidden = false;
    input.setAttribute("aria-expanded", "true");
    render(q);
  };

  const close = () => {
    list.hidden = true;
    input.setAttribute("aria-expanded", "false");
    input.value = sentenceCase(state.type.request_type);
  };

  const choose = (it) => {
    state.type = it;
    syncQuick();
    close();
    input.blur();
    changed();
  };

  const syncQuick = () => {
    for (const b of quick.children) b.classList.toggle("is-active", b.dataset.type === state.type.request_type);
  };

  input.value = sentenceCase(state.type.request_type);
  input.addEventListener("focus", () => {
    input.value = "";
    open();
  });
  input.addEventListener("input", open);
  input.addEventListener("blur", close);
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      if (!matches.length) return;
      active = (active + (e.key === "ArrowDown" ? 1 : -1) + matches.length) % matches.length;
      render(input.value.trim().toLowerCase());
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (active >= 0) choose(matches[active]);
    } else if (e.key === "Escape") {
      input.blur();
    }
  });
  list.addEventListener("mousedown", (e) => {
    const li = e.target.closest("li[data-i]");
    if (!li) return;
    e.preventDefault();          // keep focus so blur doesn't close before the pick lands
    choose(matches[Number(li.dataset.i)]);
  });

  quick.innerHTML = items.slice(0, 6).map((it) =>
    `<button type="button" class="chip" data-type="${esc(it.request_type)}">${esc(sentenceCase(it.request_type))}</button>`,
  ).join("");
  quick.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-type]");
    if (b) choose(items.find((it) => it.request_type === b.dataset.type));
  });
  syncQuick();
}

function setupDistrict() {
  const select = $("#district");
  const districts = [...state.options.districts].sort((a, b) => a.district - b.district);
  select.innerHTML = `<option value="">Not sure / anywhere in SF</option>` + districts.map((d) =>
    `<option value="${d.district}">District ${d.district} · ${esc(d.neighborhood)}</option>`).join("");
  select.value = state.district;
  select.addEventListener("change", () => {
    state.district = select.value;
    changed();
  });
}

function setupSource() {
  const group = $("#source");
  group.innerHTML = `<span class="seg-pill" aria-hidden="true"></span>` + state.options.sources.map((s) =>
    `<button type="button" role="radio" data-source="${esc(s)}">${esc(SOURCE_LABELS[s] ?? sentenceCase(s))}</button>`,
  ).join("");
  const pill = group.querySelector(".seg-pill");

  const sync = () => {
    for (const b of group.querySelectorAll("button")) {
      const on = b.dataset.source === state.source;
      b.setAttribute("aria-checked", on);
      b.tabIndex = on ? 0 : -1;
      if (on) {
        pill.style.width = `${b.offsetWidth}px`;
        pill.style.height = `${b.offsetHeight}px`;
        pill.style.transform = `translate(${b.offsetLeft}px, ${b.offsetTop}px)`;
      }
    }
  };
  const pick = (s) => {
    state.source = s;
    sync();
    changed();
  };

  group.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-source]");
    if (b) pick(b.dataset.source);
  });
  group.addEventListener("keydown", (e) => {
    const dir = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (!dir) return;
    e.preventDefault();
    const all = state.options.sources;
    const next = all[(all.indexOf(state.source) + dir + all.length) % all.length];
    pick(next);
    group.querySelector(`[data-source="${CSS.escape(next)}"]`).focus();
  });
  new ResizeObserver(sync).observe(group);
  sync();
}

function setupPhoto() {
  const box = $("#photo");
  box.checked = state.photo;
  box.addEventListener("change", () => {
    state.photo = box.checked;
    changed();
  });
}

function setupWhen() {
  const input = $("#when"), now = $("#now");
  const sync = () => {
    input.value = (state.when ?? localISO(new Date())).slice(0, 16);
    now.classList.toggle("is-active", state.when == null);
  };
  input.addEventListener("change", () => {
    state.when = input.value ? `${input.value}:00` : null;
    sync();
    changed();
  });
  now.addEventListener("click", () => {
    state.when = null;
    sync();
    changed();
  });
  sync();
}

function setupShare() {
  const btn = $("#share");
  btn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(location.href);
      btn.dataset.done = "Link copied";
    } catch {
      btn.dataset.done = "Copy failed";
    }
    btn.classList.add("is-done");
    setTimeout(() => btn.classList.remove("is-done"), 1600);
  });
}

// ---------------------------------------------------------------- prediction

let chart, timer, inflight;

function schedulePredict() {
  clearTimeout(timer);
  timer = setTimeout(predict, 120);
}

function requestBody() {
  const district = state.district === "" ? null : Number(state.district);
  return {
    opened: state.when ?? localISO(new Date()),
    category: state.type.category,
    request_type: state.type.request_type,
    request_details: null,
    source: state.source,
    supervisor_district: district,
    has_photo: state.photo,
    // options.backlog only lists non-empty queues, so a missing key means nothing is open
    backlog_at_filing: district == null ? null
      : (state.options.backlog[`${state.type.request_type}|${district}`] ?? 0),
  };
}

async function predict() {
  inflight?.abort();
  const ctrl = new AbortController();
  inflight = ctrl;
  const card = $("#result");
  card.classList.add("is-loading");
  try {
    const res = await backend.predict(requestBody(), ctrl.signal);
    if (!ctrl.signal.aborted) showResult(res);
  } catch (err) {
    if (err.name !== "AbortError") showError(err);
  } finally {
    if (inflight === ctrl) {
      inflight = null;
      card.classList.remove("is-loading");
    }
  }
}

function showResult(res) {
  $("#result").classList.remove("is-error");
  const closed = closedOnGrid(res.hours, res.survival);
  chart.update(closed);
  tweenMedian(res.median_hours ?? Infinity);

  const q25 = quantile(closed, 0.25), q75 = quantile(closed, 0.75);
  $("#range").innerHTML = Number.isFinite(q75)
    ? `Half of similar requests close between <strong>${duration(q25)}</strong> and <strong>${duration(q75)}</strong>.`
    : `Many similar requests stay open for a very long time.`;

  for (const [i, [hours]] of MILESTONES.entries()) {
    const p = valueAt(closed, hours);
    const li = $("#milestones").children[i];
    li.querySelector(".m-fill").style.transform = `scaleX(${p})`;
    li.querySelector(".m-val").textContent = pct(p);
  }

  const asOf = new Date(state.options.as_of.replace(" ", "T"));
  $("#footnote").textContent =
    `Based on ${state.type.cases.toLocaleString()} past “${sentenceCase(state.type.request_type)}” requests` +
    (Number.isNaN(asOf.getTime()) ? "." : `, data through ${asOf.toLocaleDateString(undefined, { dateStyle: "medium" })}.`);
}

function showError(err) {
  console.error(err);
  $("#result").classList.add("is-error");
  $("#range").textContent = "Couldn't get an estimate right now. Try again in a moment.";
}

let shownMedian = null, medianRaf;

function renderMedian(h) {
  const { num, unit } = durationParts(h);
  $("#median").innerHTML = `<span class="num">${num}</span><span class="unit">${unit}</span>`;
}

function tweenMedian(target) {
  cancelAnimationFrame(medianRaf);
  const from = shownMedian;
  shownMedian = target;
  if (reduceMotion || from == null || !Number.isFinite(from) || !Number.isFinite(target)) {
    renderMedian(target);
    return;
  }
  const t0 = performance.now(), dur = 650, a = Math.log(from), b = Math.log(target);
  const frame = (now) => {
    const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    renderMedian(Math.exp(a + (b - a) * e));
    if (k < 1) medianRaf = requestAnimationFrame(frame);
  };
  medianRaf = requestAnimationFrame(frame);
}

function buildMilestones() {
  $("#milestones").innerHTML = MILESTONES.map(([, label]) => `
    <li>
      <span class="m-label">Within ${label}</span>
      <span class="m-bar"><span class="m-fill"></span></span>
      <span class="m-val">–</span>
    </li>`).join("");
}

// ---------------------------------------------------------------- insights

const CATEGORICAL = ["category", "request_type", "request_details", "source", "supervisor_district", "day_of_week"];
const NUMERIC_LABELS = {
  has_photo: "Photo attached",
  log_backlog: "Longer open queue",
  backlog_missing: "Queue size unknown",
  hour_sin: "Time of day (cycle a)",
  hour_cos: "Time of day (cycle b)",
  month_sin: "Season (cycle a)",
  month_cos: "Season (cycle b)",
  night_filing: "Filed midnight–6am",
};

function featureLabel(feature) {
  const [kind, rest] = feature.includes("__") ? feature.split("__", 2) : ["num", feature];
  if (kind !== "cat") return NUMERIC_LABELS[rest] ?? sentenceCase(rest.replaceAll("_", " "));
  const col = CATEGORICAL.find((c) => rest.startsWith(`${c}_`));
  if (!col) return sentenceCase(rest.replaceAll("_", " "));
  const level = rest.slice(col.length + 1);
  if (level === "infrequent_sklearn") return `Rare ${col.replaceAll("_", " ")}`;
  if (col === "supervisor_district") {
    const d = state.options.districts.find((x) => String(x.district) === level);
    return d ? `${d.neighborhood} (D${level})` : `District ${level}`;
  }
  if (col === "source") return `Reported via ${SOURCE_LABELS[level]?.toLowerCase() ?? level}`;
  return sentenceCase(level);
}

function rateText(ratio) {
  if (Math.abs(ratio - 1) < 0.005) return "no effect";
  return ratio > 1 ? `${Math.round((ratio - 1) * 100)}% faster` : `${Math.round((1 - ratio) * 100)}% slower`;
}

// Diverging bar around ratio = 1, on a log scale so ×2 and ×½ are equally long.
function ratioBar(ratio, maxLog) {
  const w = Math.min(1, Math.abs(Math.log(ratio)) / maxLog) * 50;
  const side = ratio >= 1 ? "pos" : "neg";
  return `<span class="div-bar"><span class="div-fill ${side}" style="--w:${w}%"></span></span>`;
}

function renderInsights(report) {
  const name = report.deployed?.name;
  const cIndex = report.holdout?.[name]?.c_index;
  const data = report.data ?? {};
  const tiles = [
    ["Model in use", MODEL_LABELS[name] ?? sentenceCase(name ?? "–")],
    ["Ranking accuracy", cIndex != null ? cIndex.toFixed(3) : "–", "C-index on held-out recent cases. 0.5 is a coin flip."],
    ["Cases analysed", data.cases != null ? data.cases.toLocaleString() : "–"],
    ["Still open at snapshot", data.share_open != null ? pct(data.share_open) : "–"],
  ];
  $("#stats").innerHTML = tiles.map(([label, value, hint]) => `
    <div class="stat card"${hint ? ` title="${esc(hint)}"` : ""}>
      <span class="stat-label">${esc(label)}</span>
      <span class="stat-value">${esc(value)}</span>
    </div>`).join("");

  $("#demo-note").hidden = !report.demo;

  const coefs = (report.linear_model?.coefficients ?? [])
    .map((c) => ({ ...c, ratio: c.hazard_ratio ?? Math.exp(c.coef) }))
    .sort((a, b) => Math.abs(b.coef) - Math.abs(a.coef))
    .slice(0, 10);
  const maxCoef = Math.max(0.05, ...coefs.map((c) => Math.abs(c.coef)));
  $("#effects").innerHTML = coefs.map((c) => `
    <li>
      <span class="eff-label">${esc(featureLabel(c.feature))}</span>
      ${ratioBar(c.ratio, maxCoef)}
      <span class="eff-val ${c.ratio >= 1 ? "pos" : "neg"}">${rateText(c.ratio)}</span>
    </li>`).join("") || `<li class="empty">No coefficients in this report.</li>`;

  const hoods = (report.neighborhoods ?? []).filter((h) => h.oe != null).sort((a, b) => b.oe - a.oe);
  const maxHood = Math.max(0.05, ...hoods.map((h) => Math.abs(Math.log(h.oe))));
  $("#hoods-card").hidden = !hoods.length;
  $("#hoods").innerHTML = hoods.map((h) => `
    <li>
      <span class="eff-label">${esc(h.neighborhood)}</span>
      ${ratioBar(h.oe, maxHood)}
      <span class="eff-val ${h.oe >= 1 ? "pos" : "neg"}">${rateText(h.oe)}</span>
    </li>`).join("");
}

// ---------------------------------------------------------------- theme & motion

function setupTheme() {
  const root = document.documentElement, btn = $("#theme-toggle");
  btn.addEventListener("click", () => {
    const current = root.dataset.theme ?? (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    const apply = () => {
      root.dataset.theme = next;
      try { localStorage.setItem("howlong-theme", next); } catch { /* storage blocked */ }
    };
    if (reduceMotion || !document.startViewTransition) return apply();

    // circular reveal growing out of the toggle button
    const r = btn.getBoundingClientRect();
    const x = r.left + r.width / 2, y = r.top + r.height / 2;
    const radius = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
    document.startViewTransition(apply).ready.then(() => {
      root.animate(
        { clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${radius}px at ${x}px ${y}px)`] },
        { duration: 600, easing: "cubic-bezier(.4,0,.2,1)", pseudoElement: "::view-transition-new(root)" },
      );
    });
  });
}

function setupReveal() {
  const els = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window) || reduceMotion) {
    els.forEach((el) => el.classList.add("is-in"));
    return;
  }
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        e.target.classList.add("is-in");
        io.unobserve(e.target);
      }
    }
  }, { rootMargin: "0px 0px -10% 0px" });
  els.forEach((el) => io.observe(el));
}

buildMilestones();
boot();
