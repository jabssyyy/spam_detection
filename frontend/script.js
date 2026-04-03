const API = "http://127.0.0.1:8000";

// ── DOM references ────────────────────────────────────────────────
const msgInput    = document.getElementById("msg-input");
const charCount   = document.getElementById("char-count");
const checkBtn    = document.getElementById("check-btn");
const resultArea  = document.getElementById("result-area");
const resultSingle= document.getElementById("result-single");
const resultComp  = document.getElementById("result-compare");
const compareTable= document.getElementById("compare-table");
const errorBanner = document.getElementById("error-banner");
const errorMsg    = document.getElementById("error-msg");
const confBar     = document.getElementById("conf-bar");

// Single-model result fields
const resPrediction = document.getElementById("res-prediction");
const resSpamProb   = document.getElementById("res-spam-prob");
const resConfidence = document.getElementById("res-confidence");
const resModel      = document.getElementById("res-model");
const resCleaned    = document.getElementById("res-cleaned");
const resultBadge   = document.getElementById("result-badge");

// ── Model pill selection ──────────────────────────────────────────
let selectedModel = "svm";

document.getElementById("model-pills").addEventListener("click", e => {
  const pill = e.target.closest(".pill");
  if (!pill) return;
  document.querySelectorAll(".pill").forEach(p => p.classList.remove("active"));
  pill.classList.add("active");
  selectedModel = pill.dataset.model;
});

// ── Character counter ─────────────────────────────────────────────
msgInput.addEventListener("input", () => {
  charCount.textContent = `${msgInput.value.length} / 2000`;
});

// ── Example messages ──────────────────────────────────────────────
const SPAM_EXAMPLE = "FREE ENTRY! You have WON our weekly prize draw! To claim your £1000 Tesco voucher call 07XXXXXXXXX NOW! T&C apply.";
const HAM_EXAMPLE  = "Hey! Are you free this weekend? We're having a small get-together at mine on Saturday evening. Let me know if you can make it!";

document.getElementById("ex-spam").addEventListener("click", () => {
  msgInput.value = SPAM_EXAMPLE;
  charCount.textContent = `${SPAM_EXAMPLE.length} / 2000`;
  msgInput.focus();
});
document.getElementById("ex-ham").addEventListener("click", () => {
  msgInput.value = HAM_EXAMPLE;
  charCount.textContent = `${HAM_EXAMPLE.length} / 2000`;
  msgInput.focus();
});

// ── Enter key shortcut (Ctrl/Cmd + Enter) ────────────────────────
msgInput.addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") checkBtn.click();
});

// ── Main: analyse button ──────────────────────────────────────────
checkBtn.addEventListener("click", analyse);

async function analyse() {
  const text = msgInput.value.trim();
  if (!text) { showError("Please enter a message first."); return; }

  setLoading(true);
  hideAll();

  try {
    if (selectedModel === "all") {
      await runCompare(text);
    } else {
      await runSingle(text, selectedModel);
    }
  } catch (err) {
    showError(err.message || "Cannot reach the API. Is the server running on port 8000?");
  } finally {
    setLoading(false);
  }
}

// ── Single model prediction ───────────────────────────────────────
async function runSingle(text, model) {
  const res  = await apiFetch("/predict", "POST", { text, model });
  const isSpam = res.label === 1;

  // Badge
  resultBadge.textContent = isSpam ? "⚠ SPAM" : "✓ HAM";
  resultBadge.className   = "result-badge " + (isSpam ? "spam" : "ham");

  // Fields
  resPrediction.textContent = isSpam ? "SPAM 🚨" : "HAM ✅";
  resSpamProb.textContent   = `${(res.spam_prob * 100).toFixed(2)}%`;
  resConfidence.textContent = `${(res.confidence * 100).toFixed(2)}%`;
  resModel.textContent      = model.replace("_", " ").toUpperCase();
  resCleaned.textContent    = res.pipeline.cleaned_text || "(empty after cleaning)";

  // Confidence bar: position = spam_prob (0→left=ham, 100→right=spam)
  confBar.className = "conf-bar-fill " + (isSpam ? "spam" : "ham");
  setTimeout(() => confBar.style.width = `${res.spam_prob * 100}%`, 50);

  resultSingle.classList.add("visible");
  resultArea.classList.add("visible");
}

// ── All-models comparison ─────────────────────────────────────────
async function runCompare(text) {
  const params = new URLSearchParams({ text });
  const res    = await apiFetch(`/predict/compare?${params}`, "GET");
  const models = res.comparison;

  compareTable.innerHTML = "";
  for (const [name, data] of Object.entries(models)) {
    const isSpam     = data.prediction === "spam";
    const barColor   = isSpam ? "#ff4757" : "#2ed573";
    const pct        = Math.round(data.spam_prob * 100);
    const label      = name.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());

    compareTable.innerHTML += `
      <div class="compare-row">
        <span class="cmp-model">${label}</span>
        <span class="cmp-badge ${isSpam ? "spam" : "ham"}">${isSpam ? "SPAM" : "HAM"}</span>
        <span class="cmp-conf">spam ${pct}%</span>
        <div class="cmp-bar-mini">
          <div class="cmp-bar-mini-fill"
               style="width:${pct}%; background:${barColor}; transition: width 0.6s ease">
          </div>
        </div>
      </div>
    `;
  }

  resultComp.classList.add("visible");
  resultArea.classList.add("visible");
}

// ── API helper ────────────────────────────────────────────────────
async function apiFetch(path, method = "GET", body = null) {
  const opts = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (body) opts.body = JSON.stringify(body);

  const response = await fetch(API + path, opts);
  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.detail || `Server error ${response.status}`);
  }
  return response.json();
}

// ── UI state helpers ──────────────────────────────────────────────
function setLoading(on) {
  checkBtn.disabled = on;
  checkBtn.classList.toggle("loading", on);
}

function hideAll() {
  resultArea.classList.remove("visible");
  resultSingle.classList.remove("visible");
  resultComp.classList.remove("visible");
  errorBanner.classList.remove("visible");
  confBar.style.width = "0%";
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorBanner.classList.add("visible");
  resultArea.classList.remove("visible");
}
