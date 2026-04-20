const examplePayload = {
  age: 55,
  platelets_k_per_ul: 180,
  ast_u_l: 42,
  alt_u_l: 35,
  albumin_g_l: 39,
  fib4: 2.1,
  apri: 0.9,
  ritis: 1.2,
  nafld: -0.3,
};

const form = document.getElementById("prediction-form");
const resultState = document.getElementById("result-state");
const fillExampleButton = document.getElementById("fill-example");

function formatZoneTitle(zone) {
  if (zone === "OUT") return "Rule-out";
  if (zone === "IN") return "Rule-in";
  return "Grey zone";
}

function renderPrediction(result) {
  const probabilityPercent = (result.probability_positive * 100).toFixed(2);
  const zoneClass = result.triage_zone === "OUT" ? "zone-out" : result.triage_zone === "IN" ? "zone-in" : "zone-grey";

  resultState.innerHTML = `
    <article class="result-card">
      <section class="result-hero ${zoneClass}">
        <p class="eyebrow">Prediction Result</p>
        <h2>${formatZoneTitle(result.triage_zone)}</h2>
        <p>${result.model_name} assigned this case to <strong>${result.triage_zone}</strong>.</p>
      </section>

      <section class="result-grid">
        <div class="metric">
          <span class="metric-label">Probability of ${result.positive_label}</span>
          <span class="metric-value">${probabilityPercent}%</span>
        </div>
        <div class="metric">
          <span class="metric-label">Triage Zone</span>
          <span class="metric-value">${result.triage_zone}</span>
        </div>
        <div class="metric">
          <span class="metric-label">Rule-out Threshold</span>
          <span class="metric-value">${result.threshold_out.toFixed(4)}</span>
        </div>
        <div class="metric">
          <span class="metric-label">Rule-in Threshold</span>
          <span class="metric-value">${result.threshold_in.toFixed(4)}</span>
        </div>
      </section>
    </article>
  `;
}

function renderError(message) {
  resultState.innerHTML = `
    <article class="error-card">
      <strong>Prediction failed.</strong>
      <p>${message}</p>
    </article>
  `;
}

function collectPayload() {
  const formData = new FormData(form);
  return Object.fromEntries(
    Array.from(formData.entries()).map(([key, value]) => [key, Number(value)]),
  );
}

if (fillExampleButton) {
  fillExampleButton.addEventListener("click", () => {
    Object.entries(examplePayload).forEach(([key, value]) => {
      const input = document.getElementById(key);
      if (input) {
        input.value = value;
      }
    });
  });
}

if (form) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (!window.modelReady) {
      renderError("Model artifacts are not available. Add a trained model to the models directory.");
      return;
    }

    resultState.innerHTML = '<p class="muted">Running prediction...</p>';

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(collectPayload()),
      });

      const payload = await response.json();

      if (!response.ok) {
        const message = payload.detail || "Unexpected server error.";
        renderError(message);
        return;
      }

      renderPrediction(payload);
    } catch (error) {
      renderError("The local API is unavailable or returned an invalid response.");
    }
  });
}
