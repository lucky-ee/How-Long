// Demo backend, used when no API is configured (or it can't be reached).
// Predictions come from the true data-generating model in src/synth.py, so the numbers
// are exactly what a perfectly fitted model would say about the synthetic data.

import { GRID } from "./chart.js";

const AS_OF = "2026-10-23 20:00";
const N_CASES = 120_000;

// category -> [request types, Weibull scale (hours), Weibull shape, share of cases]
const CATEGORIES = {
  "street and sidewalk cleaning": [["bulky items", "general cleaning", "human or animal waste"], 30.0, 1.3, 0.40],
  "graffiti": [["graffiti on building commercial", "graffiti on pole"], 240.0, 0.8, 0.18],
  "encampments": [["encampment reports"], 60.0, 1.0, 0.12],
  "parking enforcement": [["blocking driveway cite only", "parking on sidewalk"], 6.0, 1.6, 0.15],
  "streetlights": [["streetlight out"], 300.0, 1.1, 0.07],
  "tree maintenance": [["tree trimming"], 700.0, 0.9, 0.08],
};

const DISTRICTS = {
  1: "Outer Richmond", 2: "Marina", 3: "North Beach", 4: "Sunset/Parkside", 5: "Haight Ashbury",
  6: "South of Market", 7: "West of Twin Peaks", 8: "Castro/Upper Market", 9: "Mission",
  10: "Bayview Hunters Point", 11: "Excelsior",
};

// True log-hazard effects. Positive = closes faster.
const EFFECTS = {
  has_photo: 0.30,
  source_mobile: 0.20,
  source_web: 0.00,
  district_10: -0.35,
  district_2: 0.25,
  night_filing: -0.25,
};

function categoryOf(requestType) {
  return Object.entries(CATEGORIES).find(([, [types]]) => types.includes(requestType));
}

function linearPredictor(body) {
  const hour = new Date(body.opened).getHours();
  return (
    EFFECTS.has_photo * Number(body.has_photo) +
    (body.source === "mobile" ? EFFECTS.source_mobile : 0) +
    (body.source === "web" ? EFFECTS.source_web : 0) +
    (body.supervisor_district === 10 ? EFFECTS.district_10 : 0) +
    (body.supervisor_district === 2 ? EFFECTS.district_2 : 0) +
    (hour < 6 ? EFFECTS.night_filing : 0)
  );
}

export const demo = {
  isDemo: true,

  async options() {
    const request_types = Object.entries(CATEGORIES).flatMap(([category, [types, , , share]]) =>
      types.map((request_type) => ({
        request_type, category, cases: Math.round((N_CASES * share) / types.length),
      })));
    return {
      as_of: AS_OF,
      request_types,
      sources: ["mobile", "phone", "web"],
      districts: Object.entries(DISTRICTS).map(([d, neighborhood]) => ({ district: Number(d), neighborhood })),
      backlog: {},
    };
  },

  async report() {
    const coefficients = [
      ["num__has_photo", EFFECTS.has_photo],
      ["cat__source_mobile", EFFECTS.source_mobile],
      ["cat__source_web", EFFECTS.source_web],
      ["cat__supervisor_district_10", EFFECTS.district_10],
      ["cat__supervisor_district_2", EFFECTS.district_2],
      ["night_filing", EFFECTS.night_filing],
    ].map(([feature, coef]) => ({ feature, coef, hazard_ratio: Math.exp(coef) }));

    const neighborhoods = Object.entries(DISTRICTS).map(([d, neighborhood]) => ({
      neighborhood, oe: Math.exp(EFFECTS[`district_${d}`] ?? 0),
    }));

    return {
      demo: true,
      data: { as_of: AS_OF, cases: N_CASES, share_open: null },
      deployed: { name: "synthetic" },
      holdout: {},
      linear_model: { coefficients },
      neighborhoods,
    };
  },

  async predict(body) {
    const found = categoryOf(body.request_type);
    if (!found) throw new Error(`Unknown request type: ${body.request_type}`);
    const [, [, scale, shape]] = found;
    const rate = Math.exp(linearPredictor(body));
    const survival = GRID.map((t) => Math.exp(-rate * Math.pow(t / scale, shape)));
    return {
      median_hours: scale * Math.pow(Math.LN2 / rate, 1 / shape),
      hours: GRID,
      survival,
    };
  },
};
