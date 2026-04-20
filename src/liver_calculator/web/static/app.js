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

function zonePresentation(zone) {
  if (zone === "OUT") {
    return {
      title: "Healthy",
      subtitle: "Low-risk interpretation",
      description: "The model placed this case in the low-risk group.",
      cssClass: "zone-out",
      badge: "Healthy",
    };
  }

  if (zone === "IN") {
    return {
      title: "Sick",
      subtitle: "High-risk interpretation",
      description: "The model placed this case in the high-risk group.",
      cssClass: "zone-in",
      badge: "Sick",
    };
  }

  return {
    title: "Grey Zone",
    subtitle: "Uncertain interpretation",
    description: "The model found this case inconclusive and closer follow-up may be needed.",
    cssClass: "zone-grey",
    badge: "Grey Zone",
  };
}

function scoreFromProbability(probability) {
  return probability * 10;
}

function thresholdCopy(score, type) {
  const rounded = score.toFixed(1);

  if (type === "low") {
    return {
      label: "Usually feels low-risk below",
      helper: `Scores lower than ${rounded}/10 usually stay in the safer range.`,
      value: `${rounded} / 10`,
    };
  }

  return {
    label: "Usually feels high-risk from",
    helper: `Scores from ${rounded}/10 and above usually move into the more concerning range.`,
    value: `${rounded} / 10`,
  };
}

function renderPrediction(result) {
  const zone = zonePresentation(result.triage_zone);
  const score = scoreFromProbability(result.probability_positive);
  const scoreValue = score.toFixed(1);
  const progressPercent = Math.min(100, Math.max(0, score * 10));
  const lowThreshold = thresholdCopy(scoreFromProbability(result.threshold_out), "low");
  const highThreshold = thresholdCopy(scoreFromProbability(result.threshold_in), "high");

  resultState.innerHTML = `
    <article class="result-card">
      <section class="result-hero ${zone.cssClass}">
        <p class="eyebrow">Prediction Result</p>
        <h2>${zone.title}</h2>
        <p class="result-subtitle">${zone.subtitle}</p>
        <p>${zone.description}</p>
      </section>

      <section class="score-card">
        <div>
          <span class="metric-label">Meta-Calculator Score</span>
          <div class="score-row">
            <span class="score-value">${scoreValue}</span>
            <span class="score-scale">/ 10</span>
          </div>
        </div>
        <span class="zone-chip ${zone.cssClass}">${zone.badge}</span>
      </section>

      <section class="score-bar-card">
        <div class="score-bar-track">
          <div class="score-bar-fill ${zone.cssClass}" style="width: ${progressPercent}%"></div>
        </div>
        <div class="score-bar-labels">
          <span>0</span>
          <span>5</span>
          <span>10</span>
        </div>
      </section>

      <section class="result-grid">
        <div class="metric">
          <span class="metric-label">${lowThreshold.label}</span>
          <span class="metric-value">${lowThreshold.value}</span>
          <p class="metric-helper">${lowThreshold.helper}</p>
        </div>
        <div class="metric">
          <span class="metric-label">${highThreshold.label}</span>
          <span class="metric-value">${highThreshold.value}</span>
          <p class="metric-helper">${highThreshold.helper}</p>
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

    resultState.innerHTML = `
      <div class="empty-card">
        <p class="eyebrow">Running prediction</p>
        <h3>Calculating result</h3>
        <p class="muted">The score and interpretation will appear here in a moment.</p>
      </div>
    `;

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
