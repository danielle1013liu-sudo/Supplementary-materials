"use strict";

let config;
let form;
let result = null;
let error = null;
let page = "home";
let collapsed = false;
let help = false;
let pending = false;
let opened = null;
let predictionRequestId = null;
let materialConstraintsOpen = false;
let screening = {
  constraints: null, result: null, error: null, requestId: null,
  transferId: null, selectedId: null, pending: false, tablePage: 0,
};
const root = document.getElementById("app");
const state = window.ScreeningState;
const tablePageSize = 20;
const experimentNames = ["macrophage_model_code", "activation_stimulus_code", "exposure_mode_code"];
const materialNames = ["hydrogel_composition_code", "bioactive_cue_code", "crosslink_method_code", "crosslink_level_code"];

function escapeHtml(value) {
  const entities = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  return String(value ?? "").replace(/[&<>"']/g, (character) => entities[character]);
}

function message(type, payload = {}) {
  window.parent.postMessage({ isStreamlitMessage: true, type, ...payload }, "*");
}

function descriptorNames() {
  return [...config.fields.map((field) => field.machine_column), ...Object.keys(config.numeric)];
}

function emptyForm() {
  return { values: Object.fromEntries(descriptorNames().map((name) => [name, ""])), unreported: {} };
}

function fieldLabel(name) {
  return config.fields.find((field) => field.machine_column === name)?.feature
    || config.numeric[name]?.display_name || name;
}

function number(value) {
  return Number(value).toLocaleString("en-GB", { maximumFractionDigits: 3 });
}

function send(action, extra = {}) {
  const id = crypto.randomUUID();
  message("streamlit:setComponentValue", {
    value: { id, action, form: structuredClone(form), ...extra }, dataType: "json",
  });
  return id;
}

function clearPrediction() {
  result = null;
  error = null;
  pending = false;
  predictionRequestId = null;
  screening.transferId = null;
  document.getElementById("result-area")?.replaceChildren();
}

function invalidate() {
  clearPrediction();
  send("change");
}

function clearScreening() {
  screening = state.invalidatedScreening(screening);
  document.getElementById("screening-result-area")?.replaceChildren();
  const summary = document.getElementById("active-constraints");
  if (summary) summary.innerHTML = constraintSummary();
}

function invalidateScreening() {
  clearScreening();
  send("screening_change", { constraints: structuredClone(screening.constraints) });
}

function navigate(next) {
  page = ["home", "prediction", "screening", "tutorial"].includes(next) ? next : "home";
  opened = null;
  help = false;
  if (innerWidth <= 700) collapsed = false;
  render();
  root.querySelector(".workspace")?.scrollTo({ top: 0, behavior: "instant" });
}

function brand() {
  return `<div><div class="brand">HY<span>DRO</span></div>
    <div class="site-name">${escapeHtml(config.name)}</div></div>`;
}

function sideLink(destination, label, icon) {
  return `<button class="side-link ${page === destination ? "selected" : ""}"
      data-page="${destination}" title="${label}">
    <span class="side-icon" aria-hidden="true">${icon}</span>
    <span class="side-label">${label}</span></button>`;
}

function sidebar() {
  return `<aside class="sidebar ${collapsed && innerWidth <= 700 ? "mobile-open" : ""}">
    <div class="side-brand"><div class="brand">HY<span>DRO</span></div>
      <button class="collapse" aria-label="${collapsed ? "Expand" : "Collapse"} sidebar">
        ${collapsed ? "›" : "‹"}</button></div>
    <section class="side-group"><h2>Dashboard</h2>
      ${sideLink("prediction", "Model Prediction", "⌁")}
      ${sideLink("screening", "Design Screening", "◇")}</section>
    <section class="side-group"><h2>Pages</h2>
      ${sideLink("home", "Landing / Home", "⌂")}
      ${sideLink("tutorial", "Tutorial", "▤")}</section>
    <p class="side-note">${escapeHtml(config.name)}<br>TNF-α and IL-10<br>
      Research predictions in pg/mL</p></aside>`;
}

function home() {
  const introduction = [
    "This tool estimates macrophage-associated TNF-α and IL-10 concentrations from hydrogel properties and experimental conditions.",
    "Select the material and culture conditions, then enter storage modulus G′, pore size and exposure time to obtain predictions from the saved models.",
    "Compare the two cytokine outputs and their feature contributions when choosing formulations for further testing.",
  ];
  return `<header class="home-nav">${brand()}<nav>
      <button class="link-button active" data-page="home">Home</button>
      <button class="link-button" data-page="prediction">Model Prediction</button>
      <button class="link-button" data-page="screening">Design Screening</button>
      <button class="link-button" data-page="tutorial">Tutorial</button>
    </nav></header><main class="home-main"><article class="home-card">
    <picture class="hero-picture">
      ${config.banner_mobile ? `<source media="(max-width:700px)" srcset="${config.banner_mobile}">` : ""}
      <img class="banner" src="${config.banner}"
        alt="Hydrogel properties and experimental conditions linked to a tree-based machine-learning model and a schematic hydrogel-delivery illustration.">
    </picture><h1 class="introduction">Introduction</h1>
    ${introduction.map((text) => `<div class="intro-row"><span class="tick" aria-hidden="true">✓</span><p>${text}</p></div>`).join("")}
    </article></main>`;
}

function category(field) {
  const name = field.machine_column;
  const current = field.categories.find((option) => option.machine_value === form.values[name]);
  const choices = opened === name
    ? `<div class="options" role="listbox" aria-label="${escapeHtml(field.feature)}">
      ${field.categories.map((option) => `<button type="button" class="option" role="option"
        aria-selected="${current?.machine_value === option.machine_value}" data-field="${name}"
        data-value="${escapeHtml(option.machine_value)}">
        ${escapeHtml(`${option.new_code} · ${option.display_label}`)}</button>`).join("")}</div>` : "";
  return `<div class="category"><label id="label-${name}">${escapeHtml(field.feature)}</label>
    <button type="button" class="choice ${current ? "" : "placeholder"}" id="${name}"
      aria-labelledby="label-${name} ${name}" aria-haspopup="listbox" aria-expanded="${opened === name}"
      data-choice="${name}">${current ? escapeHtml(`${current.new_code} · ${current.display_label}`) : `Choose ${escapeHtml(field.feature.toLowerCase())}`}</button>
    ${choices}</div>`;
}

function numeric(name, details) {
  const missing = Boolean(form.unreported[name]);
  const value = form.values[name];
  const numericValue = Number(value);
  const outside = value !== "" && Number.isFinite(numericValue)
    && (numericValue < details.minimum || numericValue > details.maximum);
  return `<section class="numeric"><div class="numeric-top">
    <label for="num-${name}">${details.display_name}</label>
    <label class="missing"><input type="checkbox" data-missing="${name}" ${missing ? "checked" : ""}>Unreported</label>
    </div><input type="range" class="slider" aria-label="${details.display_name} slider"
      data-slider="${name}" min="${details.minimum}" max="${details.maximum}" step="any"
      value="${value === "" ? details.minimum : numericValue}" ${missing ? "disabled" : ""}>
    <div class="number-wrap"><input id="num-${name}" data-number="${name}" type="text"
      inputmode="decimal" aria-label="${details.display_name} precise value" value="${escapeHtml(value)}"
      placeholder="${missing ? "Unreported" : "Enter a value"}" ${missing ? "disabled" : ""}>
      <span class="unit">${details.units}</span></div>
    <small class="range-note ${outside ? "outside" : ""}" id="range-${name}">
      ${outside ? "Typed value retained outside the slider display range. " : ""}
      Slider display: ${number(details.minimum)}–${number(details.maximum)} ${details.units}.</small></section>`;
}

function prediction() {
  return `<article class="card"><h1 class="card-header">Model Prediction</h1><div class="form-body">
    <button type="button" class="reset">Reset All</button><form id="prediction-form" novalidate>
      <div class="categories">${config.fields.map(category).join("")}</div>
      ${Object.entries(config.numeric).map(([name, details]) => numeric(name, details)).join("")}
      <div class="predict-row"><button class="predict" type="submit" ${pending ? "disabled" : ""}>
        ${pending ? "Calculating…" : "Predict"}</button></div></form>
    <div id="result-area" aria-live="polite">${results()}</div></div></article>`;
}

function downloadLink(label, content, filename, mime = "text/csv") {
  if (content === undefined || content === null) return "";
  return `<a class="download" download="${escapeHtml(filename)}"
    href="data:${mime};charset=utf-8,${encodeURIComponent(content)}">${label}</a>`;
}

function predictionCard(target) {
  return `<article class="result-card"><h3>${target.target === "tnf" ? "TNF-α" : "IL-10"}</h3>
    <div class="estimate"><span>${number(target.mean)} ± ${number(target.sd)}</span> <small>pg/mL</small></div>
    <p class="estimate-label">Mean ± SD</p>
    <div class="domain ${target.domain_pass ? "" : "outside"}">
      ${target.domain_pass ? "Within" : "Outside"} the training-data applicability domain<br>
      Domain ratio: ${number(target.domain_ratio)} (boundary 1)</div>
    ${target.invalid ? '<div class="alert">A negative concentration was returned. The unmodified values are shown.</div>' : ""}
    </article>`;
}

function contributions(target) {
  return `<div><h3>${target.target === "tnf" ? "TNF-α" : "IL-10"}</h3>
    <table class="shap-table"><tbody>${target.contributions.map((item) => `<tr>
      <td>${escapeHtml(fieldLabel(item.descriptor))}</td><td>${number(item.SHAP_pg_ml)} pg/mL</td>
      </tr>`).join("")}</tbody></table></div>`;
}

function results() {
  if (error) return `<div class="alert" role="alert">${escapeHtml(error)}</div>`;
  if (!result) return "";
  const warnings = [...new Set(result.warnings.map((warning) => warning.replace(/ seed \d+:/g, ":")))];
  return `<section class="results" id="prediction-results"><h2>Predicted cytokine responses</h2>
    <div class="result-grid">${result.targets.map(predictionCard).join("")}</div>
    ${warnings.length ? `<div class="alert">${warnings.map(escapeHtml).join("<br>")}</div>` : ""}
    <div class="downloads">
      ${downloadLink("Download predictions", result.downloads.summary_csv || result.downloads.predictions_csv, "hydrogel_predictions.csv")}
      ${downloadLink("Download inputs", result.downloads.inputs_csv, "hydrogel_inputs.csv")}
      ${downloadLink("Download contributions", result.downloads.contributions_csv, "hydrogel_contributions.csv")}
    </div><details class="shap"><summary>Feature contributions (SHAP)</summary>
      <div class="shap-grid">${result.targets.map(contributions).join("")}</div></details></section>`;
}

function screeningCategory(name) {
  const choices = config.screening_options.categorical[name] || [];
  return `<label class="screening-control" for="screen-${name}">${escapeHtml(fieldLabel(name))}
    <select id="screen-${name}" data-constraint="${name}"><option value="">Unrestricted</option>
      ${choices.map((option) => `<option value="${escapeHtml(option.value)}"
        ${screening.constraints[name] === option.value ? "selected" : ""}>
        ${escapeHtml(`${option.value} · ${option.label}`)}</option>`).join("")}</select></label>`;
}

function timeConstraint() {
  return `<label class="screening-control" for="screen-exposure_time_h">Exposure time (h)
    <select id="screen-exposure_time_h" data-time-constraint><option value="">Unrestricted</option>
      ${config.screening_options.exposure_times.map((value) => `<option value="${value}"
        ${screening.constraints.exposure_time_h === value ? "selected" : ""}>${escapeHtml(value)}</option>`).join("")}
    </select></label>`;
}

function rangeConstraint(name) {
  const limits = config.screening_options.numeric[name];
  const values = screening.constraints[name] || [null, null];
  const units = config.numeric[name].units;
  return `<fieldset class="screening-range"><legend>${escapeHtml(fieldLabel(name))} (${units})</legend>
    <div class="screening-bounds">${["Minimum", "Maximum"].map((label, index) => `<label
      for="screen-${name}-${index}">${label}<input id="screen-${name}-${index}" type="text"
      inputmode="decimal" data-range-constraint="${name}" data-bound="${index}"
      value="${escapeHtml(values[index] ?? "")}" placeholder="Unrestricted"></label>`).join("")}</div>
    <small>Existing grid: ${escapeHtml(limits.min)}–${escapeHtml(limits.max)} ${units}.</small></fieldset>`;
}

function constraintSummary() {
  const selected = [];
  for (const name of [...experimentNames, ...materialNames]) {
    const value = screening.constraints[name];
    if (value == null) continue;
    const label = config.screening_options.categorical[name]?.find((option) => option.value === value)?.label;
    selected.push(`${fieldLabel(name)}: ${label || value}`);
  }
  if (screening.constraints.exposure_time_h !== null) {
    selected.push(`Exposure time: ${screening.constraints.exposure_time_h} h`);
  }
  for (const name of ["Gprime_kPa", "pore_size_um"]) {
    const [lower, upper] = screening.constraints[name] || [null, null];
    if (lower !== null || upper !== null) {
      selected.push(`${fieldLabel(name)}: ${lower ?? "no lower bound"} to ${upper ?? "no upper bound"} ${config.numeric[name].units}`);
    }
  }
  return `<h2>Active constraints</h2><p class="setting-label">${state.experimentalSetting(screening.constraints)}</p>
    ${selected.length ? `<ul>${selected.map((text) => `<li>${escapeHtml(text)}</li>`).join("")}</ul>` : "<p>All experimental and material descriptors are unrestricted.</p>"}`;
}

function designScreening() {
  if (!config.screening_options) {
    return '<article class="card"><h1 class="card-header">Design Screening</h1><div class="form-body"><p role="status">The candidate library is not available.</p></div></article>';
  }
  return `<article class="card"><h1 class="card-header">Design Screening</h1><div class="form-body">
    <p class="screening-intro">Screen existing candidate profiles under your experimental and material constraints.
      The retained profiles balance lower predicted TNF-α with higher predicted IL-10.</p>
    <button type="button" class="reset screening-reset">Reset constraints</button>
    <form id="screening-form" novalidate><fieldset class="screening-section"><legend>Experimental conditions</legend>
      <div class="screening-controls">${experimentNames.map(screeningCategory).join("")}${timeConstraint()}</div></fieldset>
      <details id="material-constraints" class="material-constraints" ${materialConstraintsOpen ? "open" : ""}>
        <summary>Optional material constraints</summary><div class="screening-controls">
          ${materialNames.map(screeningCategory).join("")}</div>
        <div class="screening-controls">${["Gprime_kPa", "pore_size_um"].map(rangeConstraint).join("")}</div></details>
      <p class="result-note">Screening uses the existing candidate grid. Range boundaries are inclusive and compared
        with stored full-precision values. Restrictions are never rounded to a grid point or relaxed.</p>
      <div class="predict-row"><button type="submit" class="predict" ${screening.pending ? "disabled" : ""}>
        ${screening.pending ? "Screening…" : "Screen candidate profiles"}</button></div></form>
    <section id="active-constraints" class="constraint-summary">${constraintSummary()}</section>
    <div id="screening-result-area" aria-live="polite">${screeningResults()}</div></div></article>`;
}

function scatterPlot(screened) {
  const candidates = screened.plot?.candidates || [];
  const retained = screened.retained || [];
  const all = [...candidates, ...retained.map((profile) => [profile.tnf_mean, profile.il10_mean])];
  if (!all.length) return "";
  const xMaximum = all.reduce((maximum, point) => Math.max(maximum, point[0]), 1) * 1.06;
  const yMaximum = all.reduce((maximum, point) => Math.max(maximum, point[1]), 1) * 1.06;
  const x = (value) => 80 + (value / xMaximum) * 650;
  const y = (value) => 355 - (value / yMaximum) * 305;
  const ticks = Array.from({ length: 5 }, (_, index) => index / 4);
  const axes = ticks.map((fraction) => `<g class="plot-tick"><line x1="${x(fraction * xMaximum)}"
      y1="355" x2="${x(fraction * xMaximum)}" y2="361"></line>
    <text x="${x(fraction * xMaximum)}" y="381" text-anchor="middle">${number(fraction * xMaximum)}</text>
    <line x1="74" y1="${y(fraction * yMaximum)}" x2="80" y2="${y(fraction * yMaximum)}"></line>
    <text x="68" y="${y(fraction * yMaximum) + 4}" text-anchor="end">${number(fraction * yMaximum)}</text></g>`).join("");
  return `<figure class="screening-figure"><svg viewBox="0 0 780 435" role="img"
      aria-label="Candidate profiles and retained profiles plotted by predicted TNF-alpha and predicted IL-10">
    <line class="plot-axis" x1="80" y1="50" x2="80" y2="355"></line>
    <line class="plot-axis" x1="80" y1="355" x2="730" y2="355"></line>${axes}
    ${candidates.map((point) => `<circle class="candidate-point" cx="${x(point[0])}" cy="${y(point[1])}" r="2.2"></circle>`).join("")}
    ${retained.map((profile) => `<circle class="retained-point" cx="${x(profile.tnf_mean)}" cy="${y(profile.il10_mean)}" r="4.5">
      <title>${escapeHtml(profile.candidate_id)}: TNF-α ${number(profile.tnf_mean)}, IL-10 ${number(profile.il10_mean)} pg/mL</title></circle>`).join("")}
    <text x="405" y="421" text-anchor="middle" class="plot-label">Predicted TNF-α (pg/mL)</text>
    <text transform="translate(19 202) rotate(-90)" text-anchor="middle" class="plot-label">Predicted IL-10 (pg/mL)</text>
    </svg><figcaption><span class="legend-point candidate-legend"></span>Candidate profiles
      <span class="legend-point retained-legend"></span>Retained profiles</figcaption>
    ${screened.plot?.sampled ? '<p class="result-note">Candidate points are sampled for display only. Selection uses every matching eligible candidate; all retained profiles are included in the table and download.</p>' : ""}</figure>`;
}

function profileValue(profile, name) {
  const value = profile.values[name];
  if (value === null) return "Unreported";
  const label = config.screening_options.categorical[name]?.find((option) => option.value === value)?.label;
  return label ? `${value} · ${label}` : String(value);
}

function retainedTable(screened) {
  const names = descriptorNames();
  const first = screening.tablePage * tablePageSize;
  const visible = screened.retained.slice(first, first + tablePageSize);
  const pages = Math.ceil(screened.retained.length / tablePageSize);
  const headings = ["Candidate ID", ...names.map((name) => `${fieldLabel(name)}${config.numeric[name] ? ` (${config.numeric[name].units})` : ""}`),
    "Predicted TNF-α (pg/mL)", "Predicted IL-10 (pg/mL)", "TNF-α AD", "IL-10 AD"];
  return `<p class="result-note">Cytokine values are shown as Mean ± SD. AD: applicability domain.</p>
    <div class="screening-table-wrap" tabindex="0" aria-label="Retained profiles; scroll horizontally for all descriptors">
    <table class="screening-table"><thead><tr>${headings.map((label) => `<th scope="col">${escapeHtml(label)}</th>`).join("")}</tr></thead>
    <tbody>${visible.map((profile) => `<tr><th scope="row">${escapeHtml(profile.candidate_id)}</th>
      ${names.map((name) => `<td>${escapeHtml(profileValue(profile, name))}</td>`).join("")}
      <td>${number(profile.tnf_mean)} ± ${number(profile.tnf_sd)}</td>
      <td>${number(profile.il10_mean)} ± ${number(profile.il10_sd)}</td>
      <td>${profile.tnf_AD_pass ? "Within AD" : "Outside AD"}</td>
      <td>${profile.il10_AD_pass ? "Within AD" : "Outside AD"}</td></tr>`).join("")}</tbody></table></div>
    ${pages > 1 ? `<div class="table-pagination"><button type="button" data-table-page="${screening.tablePage - 1}"
      ${screening.tablePage === 0 ? "disabled" : ""}>Previous</button>
      <span>Profiles ${first + 1}–${Math.min(first + tablePageSize, screened.retained.length)} of ${screened.retained.length}</span>
      <button type="button" data-table-page="${screening.tablePage + 1}" ${screening.tablePage + 1 === pages ? "disabled" : ""}>Next</button></div>` : ""}`;
}

function screeningResults() {
  if (screening.error) return `<div class="alert" role="alert">${escapeHtml(screening.error)}</div>`;
  const screened = screening.result;
  if (!screened) return "";
  const counts = `<div class="screening-counts"><div><strong>${number(screened.matching_count)}</strong>
    <span>Matching eligible candidates</span></div><div><strong>${number(screened.retained_count)}</strong>
    <span>Retained profiles</span></div></div>`;
  const downloads = `<div class="downloads">
    ${downloadLink("Download retained profiles", screened.downloads?.profiles_csv, "hydrogel_retained_profiles.csv")}
    ${downloadLink("Download active constraints", screened.downloads?.constraints_json, "hydrogel_screening_constraints.json", "application/json")}</div>`;
  if (!screened.matching_count) {
    return `<section class="results">${counts}<p>No eligible candidates match the selected conditions.</p>${downloads}</section>`;
  }
  const table = screened.retained_count ? retainedTable(screened) :
    '<p>No profiles satisfy the consistency criterion under the selected conditions. The criterion has not been relaxed.</p>';
  const selection = screened.retained_count ? `<div class="candidate-transfer">
    <label for="selected-profile">Inspect a retained profile</label>
    <select id="selected-profile"><option value="">Choose a candidate profile</option>
      ${screened.retained.map((profile) => `<option value="${escapeHtml(profile.candidate_id)}"
        ${screening.selectedId === profile.candidate_id ? "selected" : ""}>${escapeHtml(profile.candidate_id)}</option>`).join("")}</select>
    <button type="button" class="predict" id="load-candidate" ${!screening.selectedId || screening.transferId ? "disabled" : ""}>
      ${screening.transferId ? "Loading…" : "Load into Model Prediction"}</button>
    <p class="result-note">The exact ten descriptor values are transferred. Use Predict to inspect saved-model outputs and feature contributions.</p></div>` : "";
  return `<section class="results" id="screening-results"><h2>Candidate profiles</h2>${counts}
    ${scatterPlot(screened)}${table}${downloads}${selection}</section>`;
}

const featureHelp = {
  hydrogel_composition_code: "Polymer origin: natural, synthetic, or hybrid.",
  bioactive_cue_code: "Material-associated biological function.",
  crosslink_method_code: "How the gel network is formed.",
  crosslink_level_code: "Low, medium or high network-density category.",
  macrophage_model_code: "Macrophage cell model or experimental context.",
  activation_stimulus_code: "Stimulation applied to the cells.",
  exposure_mode_code: "How cells or tissues interact with the hydrogel.",
  Gprime_kPa: "Reported storage modulus G′, in kPa. Select Unreported when no storage-modulus measurement is available.",
  pore_size_um: "Reported characteristic pore size.",
  exposure_time_h: "Duration of culture or material exposure.",
};

function helpPanel() {
  const fields = [...config.fields.map((field) => [field.machine_column, field.feature]),
    ...Object.entries(config.numeric).map(([name, details]) => [name, `${details.display_name} (${details.units})`])];
  return `<section class="help-panel" aria-label="Descriptor help">
    <button class="close" aria-label="Close descriptor help">×</button><h2>Features</h2>
    ${fields.map(([name, label]) => `<h3>${escapeHtml(label)}</h3><p>${featureHelp[name]}</p>`).join("")}
    <a download="${escapeHtml(config.codebook_filename)}" href="${config.codebook}">Download descriptor codebook</a></section>`;
}

function tutorial() {
  const steps = [
    ["Choose your experiment", "Select the hydrogel properties, macrophage model, activation stimulus and exposure mode."],
    ["Enter measured values", "Add storage modulus G′ (kPa), pore size (µm) and exposure time (h). Select Unreported for a missing numeric value."],
    ["Predict responses", "Click Predict to view TNF-α and IL-10 concentrations in pg/mL, reported as mean ± SD."],
    ["Compare and download", "Review feature contributions, adjust the inputs, and download your inputs and results. Use Reset All to start again."],
    ["Screen candidate profiles", "Open Design Screening to restrict experimental conditions and optional material properties. Screen the existing grid, review the retained profiles, and load a selected profile into Model Prediction."],
  ];
  return `<article class="card"><h1 class="card-header">Tutorial</h1><div class="tutorial-body">
    ${steps.map(([title, text], index) => `<section class="tutorial-step"><h2>${index + 1}. ${title}</h2><p>${text}</p></section>`).join("")}
    <button class="predict" data-page="prediction">Start predicting</button></div></article>`;
}

function bindPredictionEvents() {
  root.querySelector(".reset:not(.screening-reset)")?.addEventListener("click", () => {
    form = emptyForm();
    clearPrediction();
    opened = null;
    send("reset");
    render();
  });
  root.querySelectorAll("[data-choice]").forEach((button) => {
    button.onclick = () => {
      opened = opened === button.dataset.choice ? null : button.dataset.choice;
      render();
      if (opened) root.querySelector(".option")?.focus();
    };
  });
  root.querySelectorAll("[data-value]").forEach((button) => {
    button.onclick = () => {
      form.values[button.dataset.field] = button.dataset.value;
      opened = null;
      invalidate();
      render();
      document.getElementById(button.dataset.field)?.focus();
    };
  });
  root.querySelectorAll("[data-number]").forEach((input) => {
    input.oninput = () => {
      form.values[input.dataset.number] = input.value;
      clearPrediction();
      const slider = root.querySelector(`[data-slider="${input.dataset.number}"]`);
      if (input.value !== "" && Number.isFinite(Number(input.value))) slider.value = Number(input.value);
    };
    input.onchange = () => {
      invalidate();
      render();
    };
  });
  root.querySelectorAll("[data-slider]").forEach((slider) => {
    slider.oninput = () => {
      form.values[slider.dataset.slider] = slider.value;
      document.getElementById(`num-${slider.dataset.slider}`).value = slider.value;
      clearPrediction();
    };
    slider.onchange = () => {
      invalidate();
      render();
    };
  });
  root.querySelectorAll("[data-missing]").forEach((input) => {
    input.onchange = () => {
      form.unreported[input.dataset.missing] = input.checked;
      invalidate();
      render();
    };
  });
  root.querySelector("#prediction-form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    opened = null;
    clearPrediction();
    pending = true;
    predictionRequestId = send("predict");
    render();
  });
}

function bindScreeningEvents() {
  root.querySelector(".screening-reset")?.addEventListener("click", () => {
    screening.constraints = state.emptyConstraints(config.screening_options);
    clearScreening();
    send("screening_reset", { constraints: structuredClone(screening.constraints) });
    render();
  });
  root.querySelector("#material-constraints")?.addEventListener("toggle", (event) => {
    materialConstraintsOpen = event.target.open;
  });
  root.querySelectorAll("[data-constraint]").forEach((select) => {
    select.onchange = () => {
      screening.constraints[select.dataset.constraint] = select.value || null;
      invalidateScreening();
      render();
    };
  });
  root.querySelector("[data-time-constraint]")?.addEventListener("change", (event) => {
    screening.constraints.exposure_time_h = event.target.value === "" ? null : Number(event.target.value);
    invalidateScreening();
    render();
  });
  root.querySelectorAll("[data-range-constraint]").forEach((input) => {
    input.oninput = () => {
      const name = input.dataset.rangeConstraint;
      screening.constraints[name] ||= [null, null];
      screening.constraints[name][Number(input.dataset.bound)] = input.value === "" ? null : input.value;
      clearScreening();
    };
    input.onchange = () => {
      invalidateScreening();
      render();
    };
  });
  root.querySelector("#screening-form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    clearScreening();
    screening.pending = true;
    screening.requestId = send("screen", { constraints: structuredClone(screening.constraints) });
    render();
  });
  root.querySelectorAll("[data-table-page]").forEach((button) => {
    button.onclick = () => {
      screening.tablePage = Number(button.dataset.tablePage);
      render();
    };
  });
  root.querySelector("#selected-profile")?.addEventListener("change", (event) => {
    screening.selectedId = event.target.value || null;
    render();
  });
  root.querySelector("#load-candidate")?.addEventListener("click", () => {
    if (!screening.selectedId) return;
    screening.transferId = send("load_candidate", {
      candidate_id: screening.selectedId,
      constraints: structuredClone(screening.constraints),
    });
    render();
  });
}

function render() {
  const scroll = root.querySelector(".workspace")?.scrollTop || 0;
  const active = document.activeElement;
  const focus = active?.id ? { id: active.id, start: active.selectionStart, end: active.selectionEnd } : null;
  root.className = collapsed && innerWidth > 700 ? "collapsed" : "";
  const content = page === "tutorial" ? tutorial() : page === "screening" ? designScreening() : prediction();
  root.innerHTML = page === "home" ? home() : `${sidebar()}<main class="workspace">${content}</main>
    <button class="help-tab" aria-label="Open descriptor help">?</button>${help ? helpPanel() : ""}`;
  root.querySelectorAll("[data-page]").forEach((button) => {
    button.onclick = () => navigate(button.dataset.page);
  });
  root.querySelector(".collapse")?.addEventListener("click", () => {
    collapsed = !collapsed;
    render();
  });
  root.querySelector(".help-tab")?.addEventListener("click", () => {
    help = true;
    render();
  });
  root.querySelector(".close")?.addEventListener("click", () => {
    help = false;
    render();
  });
  bindPredictionEvents();
  bindScreeningEvents();
  const workspace = root.querySelector(".workspace");
  if (workspace) workspace.scrollTop = scroll;
  const restored = focus && document.getElementById(focus.id);
  if (restored) {
    restored.focus({ preventScroll: true });
    if (typeof restored.setSelectionRange === "function" && focus.start !== null) {
      restored.setSelectionRange(focus.start, focus.end);
    }
  }
}

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    opened = null;
    help = false;
    render();
  }
});

window.addEventListener("message", (event) => {
  if (event.source !== window.parent || event.data.type !== "streamlit:render") return;
  const previousVersion = config?.library_version;
  const initialRender = !config;
  config = event.data.args;
  if (!form) form = config.form || emptyForm();
  if (!screening.constraints) {
    screening.constraints = {
      ...state.emptyConstraints(config.screening_options),
      ...(config.screening_constraints || {}),
    };
  }
  if (previousVersion && previousVersion !== config.library_version) {
    screening.constraints = state.emptyConstraints(config.screening_options);
    screening = state.invalidatedScreening(screening);
  }
  if (initialRender) {
    result = config.result || null;
    error = config.error || null;
    screening.result = config.screening_result || null;
    screening.error = config.screening_error || null;
  }
  if (state.acceptResponse(predictionRequestId, config.prediction_response_id)) {
    result = config.result;
    error = config.error;
    pending = false;
  }
  if (state.acceptResponse(screening.requestId, config.screening_response_id)) {
    screening.result = config.screening_result;
    screening.error = config.screening_error;
    screening.pending = false;
  }
  if (state.acceptResponse(screening.transferId, config.transfer_response_id)) {
    screening.transferId = null;
    if (config.screening_error) {
      screening.error = config.screening_error;
    } else {
      form = state.profileForm(config.form.values, descriptorNames());
      clearPrediction();
      page = "prediction";
    }
  }
  render();
});

function height() {
  let frameHeight = 900;
  try {
    frameHeight = window.parent.innerHeight;
  } catch {
    // Cross-origin embedding uses the fallback height.
  }
  message("streamlit:setFrameHeight", { height: frameHeight });
}

window.addEventListener("resize", height);
try {
  window.parent.addEventListener("resize", height);
  window.addEventListener("unload", () => window.parent.removeEventListener("resize", height));
} catch {
  // The local resize listener remains available across origins.
}
message("streamlit:componentReady", { apiVersion: 1 });
height();
