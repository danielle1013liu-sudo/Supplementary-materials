(function (global) {
  "use strict";
  const experimentFields = [
    "macrophage_model_code", "activation_stimulus_code", "exposure_mode_code", "exposure_time_h",
  ];

  function emptyConstraints(options) {
    return {
      ...Object.fromEntries(Object.keys(options?.categorical || {}).map((name) => [name, null])),
      exposure_time_h: null,
      Gprime_kPa: [null, null],
      pore_size_um: [null, null],
    };
  }

  function experimentalSetting(constraints) {
    const count = experimentFields.filter((name) => constraints[name] != null).length;
    if (count === experimentFields.length) return "Fully fixed experimental setting";
    return count ? "Partially constrained experimental setting" : "Unrestricted experimental setting";
  }

  function acceptResponse(requestId, responseId) {
    return Boolean(requestId && responseId && requestId === responseId);
  }

  function invalidatedScreening(state) {
    return { ...state, result: null, error: null, requestId: null, transferId: null,
      selectedId: null, pending: false, tablePage: 0 };
  }

  function profileForm(values, descriptorNames) {
    if (!values || descriptorNames.some((name) => !(name in values))) {
      throw new Error("The selected profile does not contain all ten descriptors.");
    }
    return {
      values: Object.fromEntries(descriptorNames.map((name) => [name, values[name]])),
      unreported: Object.fromEntries(["Gprime_kPa", "pore_size_um", "exposure_time_h"]
        .map((name) => [name, values[name] === null])),
    };
  }

  const helpers = { emptyConstraints, experimentalSetting, acceptResponse, invalidatedScreening, profileForm };
  if (typeof module !== "undefined" && module.exports) module.exports = helpers;
  global.ScreeningState = helpers;
})(typeof window !== "undefined" ? window : globalThis);
