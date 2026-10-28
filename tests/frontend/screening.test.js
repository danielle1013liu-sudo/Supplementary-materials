const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const helpers = require("../../web/screening-state.js");

const categories = [
  "hydrogel_composition_code", "bioactive_cue_code", "crosslink_method_code", "crosslink_level_code",
  "macrophage_model_code", "activation_stimulus_code", "exposure_mode_code",
];
const numeric = {
  Gprime_kPa: { display_name: "Storage modulus G′", units: "kPa", minimum: 0, maximum: 100 },
  pore_size_um: { display_name: "Pore size", units: "µm", minimum: 0, maximum: 500 },
  exposure_time_h: { display_name: "Exposure time", units: "h", minimum: 0, maximum: 72 },
};
const names = [...categories, ...Object.keys(numeric)];
const options = {
  categorical: Object.fromEntries(categories.map((name) => [name, [{ value: "A", label: "Category A" }]])),
  numeric: { Gprime_kPa: { min: 0.12345678901234566, max: 100 }, pore_size_um: { min: 1, max: 500 } },
  exposure_times: [24, 48, 72],
};

function harness() {
  const listeners = new Map();
  const messages = [];
  const root = {
    innerHTML: "", className: "",
    querySelector: () => null,
    querySelectorAll: () => [],
  };
  const parent = {
    innerHeight: 900,
    postMessage: (message) => messages.push(message),
    addEventListener: () => {},
    removeEventListener: () => {},
  };
  const window = {
    parent,
    addEventListener: (name, callback) => listeners.set(name, callback),
  };
  const document = { activeElement: null, getElementById: (id) => id === "app" ? root : null };
  let nextId = 0;
  const context = vm.createContext({
    window, document, innerWidth: 1440, structuredClone,
    crypto: { randomUUID: () => `event-${++nextId}` }, console,
  });
  for (const file of ["screening-state.js", "app.js"]) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, "../../web", file), "utf8"), context);
  }
  const config = {
    name: "Hydrogel Cytokine Explorer", library_version: "library-v1", numeric,
    fields: categories.map((name) => ({ machine_column: name, feature: name,
      categories: [{ machine_value: "A", new_code: "A", display_label: "Category A" }] })),
    screening_options: options, banner: "data:image/svg+xml,test", codebook: "data:application/json,test",
    codebook_filename: "hydrogel_descriptor_codebook.json", form: null,
  };
  function render(args = {}) {
    listeners.get("message")({ source: parent, data: { type: "streamlit:render", args: { ...config, ...args } } });
  }
  render();
  return { context, messages, root, render, run: (source) => vm.runInContext(source, context) };
}

function plain(value) {
  return JSON.parse(JSON.stringify(value));
}

test("experimental setting distinguishes unrestricted, partial and fully fixed searches", () => {
  const constraints = helpers.emptyConstraints(options);
  assert.equal(helpers.experimentalSetting(constraints), "Unrestricted experimental setting");
  constraints.macrophage_model_code = "A";
  assert.equal(helpers.experimentalSetting(constraints), "Partially constrained experimental setting");
  constraints.activation_stimulus_code = "A";
  constraints.exposure_mode_code = "A";
  constraints.exposure_time_h = 24;
  assert.equal(helpers.experimentalSetting(constraints), "Fully fixed experimental setting");
});

test("profile transfer preserves ten exact values and explicit missing values", () => {
  const values = { ...Object.fromEntries(categories.map((name) => [name, "A"])),
    Gprime_kPa: 0.12345678901234566, pore_size_um: null, exposure_time_h: 24 };
  const form = helpers.profileForm(values, names);
  assert.deepEqual(form.values, values);
  assert.equal(form.unreported.pore_size_um, true);
  assert.equal(form.unreported.Gprime_kPa, false);
  assert.throws(() => helpers.profileForm({ Gprime_kPa: 2 }, names), /all ten descriptors/);
  form.values.Gprime_kPa = 4;
  assert.equal(values.Gprime_kPa, 0.12345678901234566);
});

test("screening responses from an earlier constraint request are ignored", () => {
  const app = harness();
  app.run('navigate("screening"); screening.pending = true; screening.requestId = send("screen", {constraints: screening.constraints});');
  const requestId = app.messages.at(-1).value.id;
  app.run('screening.constraints.Gprime_kPa = ["0.12345678901234566", "9.999999999999998"]; invalidateScreening(); render();');
  app.render({ screening_response_id: requestId,
    screening_result: { matching_count: 0, retained_count: 0, retained: [] } });
  assert.equal(app.run("screening.result"), null);
  assert.equal(app.run("screening.pending"), false);
  assert.equal(app.run("screening.requestId"), null);
  assert.ok(!app.root.innerHTML.includes("No eligible candidates match"));
  assert.ok(app.root.innerHTML.includes("0.12345678901234566 to 9.999999999999998"));
  assert.deepEqual(plain(app.messages.at(-1).value.constraints.Gprime_kPa),
    ["0.12345678901234566", "9.999999999999998"]);
});

test("empty matching sets show the specified message without substituting conditions", () => {
  const app = harness();
  app.run('navigate("screening"); screening.requestId = send("screen", {constraints: screening.constraints});');
  app.render({ screening_response_id: app.messages.at(-1).value.id,
    screening_result: { matching_count: 0, retained_count: 0, retained: [] } });
  assert.ok(app.root.innerHTML.includes("No eligible candidates match the selected conditions."));
  assert.ok(app.root.innerHTML.includes("Unrestricted experimental setting"));
});

test("only the active candidate-transfer response can replace the prediction form", () => {
  const app = harness();
  const values = { ...Object.fromEntries(categories.map((name) => [name, "A"])),
    Gprime_kPa: 0.12345678901234566, pore_size_um: 177.8234567890123, exposure_time_h: 24 };
  app.run('screening.transferId = send("load_candidate", {candidate_id: "C0000001", constraints: screening.constraints});');
  const id = app.messages.at(-1).value.id;
  app.render({ transfer_response_id: "older-transfer", form: { values, unreported: {} } });
  assert.equal(app.run("page"), "home");
  app.render({ transfer_response_id: id, form: { values, unreported: {} } });
  assert.equal(app.run("page"), "prediction");
  assert.deepEqual(plain(app.run("form.values")), values);
  assert.equal(app.run("result"), null);
  assert.ok(app.root.innerHTML.includes('value="0.12345678901234566"'));
  assert.ok(app.root.innerHTML.includes('value="177.8234567890123"'));
});

test("editing a prediction cancels a pending profile transfer and stale prediction results", () => {
  const app = harness();
  app.run('predictionRequestId = "old-prediction"; screening.transferId = "old-transfer"; clearPrediction();');
  app.render({ prediction_response_id: "old-prediction", result: { targets: [] },
    transfer_response_id: "old-transfer", form: { values: {}, unreported: {} } });
  assert.equal(app.run("result"), null);
  assert.equal(app.run("screening.transferId"), null);
  assert.equal(app.run("page"), "home");
});

test("resetting screening invalidates the previous result and candidate selection", () => {
  const state = helpers.invalidatedScreening({ constraints: { exposure_time_h: 24 },
    result: { retained_count: 1 }, error: "old error", requestId: "old", transferId: "transfer",
    selectedId: "C0000001", pending: true, tablePage: 3 });
  assert.equal(state.result, null);
  assert.equal(state.error, null);
  assert.equal(state.requestId, null);
  assert.equal(state.transferId, null);
  assert.equal(state.selectedId, null);
  assert.equal(state.pending, false);
  assert.equal(state.tablePage, 0);
});
