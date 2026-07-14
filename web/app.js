const $ = (id) => document.getElementById(id);

const fmtInt = (n) => Math.round(n).toLocaleString("en-US");

const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function fmtParams(n) {
  if (!n) return "—";
  if (n >= 1e9) return (n / 1e9).toFixed(1) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(Math.round(n));
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

const easeOut = (t) => 1 - Math.pow(1 - t, 3);

/** Ticks a number up to its target instead of snapping to it. */
function countUp(el, target, format = fmtInt, duration = 1100) {
  if (!target || REDUCED) {
    el.textContent = target ? format(target) : "—";
    return;
  }
  const t0 = performance.now();
  const step = (now) => {
    const p = Math.min((now - t0) / duration, 1);
    el.textContent = format(target * easeOut(p));
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* ---------------- model info ---------------- */

let MODEL = null;

async function loadModel() {
  MODEL = await (await fetch("/api/model")).json();
  const c = MODEL.config || {};

  countUp($("stat-params"), MODEL.params, fmtParams);
  countUp($("stat-layers"), c.num_layers);
  countUp($("stat-heads"), c.num_heads);
  countUp($("stat-dmodel"), c.d_model);
  countUp($("stat-ctx"), c.max_len);
  countUp($("stat-step"), MODEL.step);

  if (MODEL.model_loaded) {
    $("arch-in").textContent = `(B, ${c.max_len})`;
    $("arch-emb").textContent = `vocab ${fmtInt(c.vocab_size)} × d_model ${c.d_model}`;
    $("arch-nlayers").textContent = c.num_layers;
    $("arch-attn").textContent = `${c.num_heads} heads × d_head ${MODEL.d_head}`;
    $("arch-mlp").textContent = `d_model ${c.d_model} → 2 × d_ff ${c.d_ff} → d_model ${c.d_model}`;
    $("arch-out").textContent = `d_model ${c.d_model} → vocab ${fmtInt(c.vocab_size)}`;
    $("arch-logits").textContent = `(B, ${c.max_len}, ${fmtInt(c.vocab_size)})`;

    $("load-state").innerHTML =
      `<span class="ok">●</span> checkpoint loaded — ${fmtInt(MODEL.params)} parameters, ` +
      `step ${fmtInt(MODEL.step)}, ${MODEL.tokenizer}`;
  } else {
    // No weights: the showcase still stands on its own, but be explicit that
    // the chat can't run rather than quietly serving noise.
    $("load-state").innerHTML =
      `<div class="warn"><b>No checkpoint loaded.</b><br>` +
      `${escapeHtml(MODEL.error || "")}<br><br>` +
      `The architecture and training sections below are static. To enable chat, ` +
      `put a checkpoint at <code>${escapeHtml(MODEL.ckpt_path)}</code> and restart the server, ` +
      `or start it with <code>python serve.py --ckpt path/to/your.ckpt</code>.</div>`;

    $("chat-text").disabled = true;
    $("chat-send").disabled = true;
    $("chat-text").placeholder = "No checkpoint loaded — chat is disabled";
    $("empty-chat").textContent = "Load a checkpoint to talk to KEVIN.";
  }

  $("footer-meta").textContent = MODEL.model_loaded
    ? `${MODEL.ckpt_path} · ${MODEL.tokenizer} · float32 numpy forward pass`
    : `no checkpoint · expected at ${MODEL.ckpt_path}`;
}

/* ---------------- charts ---------------- */

const CSS = getComputedStyle(document.documentElement);
const COLOR = {
  line: CSS.getPropertyValue("--line").trim(),
  faint: CSS.getPropertyValue("--text-faint").trim(),
  accent: CSS.getPropertyValue("--accent").trim(),
  green: CSS.getPropertyValue("--accent-green").trim(),
};

/** progress 0..1 draws the line in from the left; axes are always fully drawn. */
function drawChart(canvas, points, { color, logScale = false, progress = 1 }) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = 300;

  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const pad = { top: 12, right: 12, bottom: 26, left: 52 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  const tx = (v) => (logScale ? Math.log10(Math.max(v, 1e-6)) : v);
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => tx(p.y));

  // Scale to the full series, not the visible slice, so the axes don't jump
  // around while the line animates in.
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (y0 === y1) { y0 -= 0.5; y1 += 0.5; }
  const yPad = (y1 - y0) * 0.08;
  y0 -= yPad; y1 += yPad;

  const px = (x) => pad.left + ((x - x0) / (x1 - x0 || 1)) * plotW;
  const py = (y) => pad.top + plotH - ((y - y0) / (y1 - y0 || 1)) * plotH;

  ctx.font = "11px ui-monospace, monospace";
  ctx.fillStyle = COLOR.faint;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";

  for (let i = 0; i <= 4; i++) {
    const yv = y0 + ((y1 - y0) * i) / 4;
    const y = py(yv);
    ctx.strokeStyle = COLOR.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();

    const label = logScale ? Math.pow(10, yv) : yv;
    ctx.fillText(label >= 1000 ? label.toExponential(0) : label.toFixed(2), pad.left - 8, y);
  }

  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let i = 0; i <= 4; i++) {
    const xv = x0 + ((x1 - x0) * i) / 4;
    ctx.fillText(Math.round(xv), px(xv), h - pad.bottom + 8);
  }

  const shown = points.slice(0, Math.max(2, Math.ceil(points.length * progress)));

  const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
  grad.addColorStop(0, color + "33");
  grad.addColorStop(1, color + "00");
  ctx.fillStyle = grad;
  ctx.beginPath();
  ctx.moveTo(px(shown[0].x), py(tx(shown[0].y)));
  shown.forEach((p) => ctx.lineTo(px(p.x), py(tx(p.y))));
  ctx.lineTo(px(shown[shown.length - 1].x), pad.top + plotH);
  ctx.lineTo(px(shown[0].x), pad.top + plotH);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = color;
  ctx.lineWidth = 1.75;
  ctx.lineJoin = "round";
  ctx.beginPath();
  shown.forEach((p, i) => {
    const X = px(p.x), Y = py(tx(p.y));
    i === 0 ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y);
  });
  ctx.stroke();

  // Glowing head on the leading edge while the line is still drawing.
  if (progress < 1) {
    const last = shown[shown.length - 1];
    const X = px(last.x), Y = py(tx(last.y));
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(X, Y, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
  }
}

let METRICS = null;
let chartsDrawn = false;

async function loadMetrics() {
  METRICS = await (await fetch("/api/metrics")).json();

  if (!METRICS.available) {
    document.querySelector(".charts").hidden = true;
    const empty = $("metrics-empty");
    empty.hidden = false;
    empty.innerHTML =
      `No training metrics yet.<br>` +
      `The trainer appends step / loss / perplexity to <code>${escapeHtml(METRICS.path)}</code> ` +
      `on every step — run <code>python train.py</code> and the curves appear here.`;
    return;
  }

  const s = METRICS.summary;
  $("metrics-summary").innerHTML = `
    <div><b>${fmtInt(s.steps)}</b>steps</div>
    <div><b>${s.final_loss.toFixed(4)}</b>final loss</div>
    <div><b>${s.best_loss.toFixed(4)}</b>best loss</div>
    <div><b>${s.final_perplexity.toFixed(1)}</b>final perplexity</div>
    <div><b>${fmtInt(s.mean_tokens_per_sec)}</b>tok/s while training</div>
    <div><b>${(s.elapsed_sec / 60).toFixed(1)}m</b>wall clock</div>`;

  renderCharts();
}

function renderCharts(progress = 1) {
  if (!METRICS?.available) return;
  drawChart($("chart-loss"), METRICS.rows.map((r) => ({ x: r.step, y: r.loss })),
            { color: COLOR.accent, progress });
  drawChart($("chart-ppl"), METRICS.rows.map((r) => ({ x: r.step, y: r.perplexity })),
            { color: COLOR.green, logScale: true, progress });
}

/** Draw the curves in once, the first time the training section scrolls in. */
function animateCharts() {
  if (chartsDrawn || !METRICS?.available) return;
  chartsDrawn = true;

  if (REDUCED) { renderCharts(1); return; }

  const t0 = performance.now();
  const duration = 1200;
  const step = (now) => {
    const p = Math.min((now - t0) / duration, 1);
    renderCharts(easeOut(p));
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

window.addEventListener("resize", () => renderCharts(1));

/* ---------------- scroll reveal, nav, spotlight ---------------- */

const revealObserver = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    e.target.classList.add("in");
    revealObserver.unobserve(e.target);
  }
}, { threshold: 0.15, rootMargin: "0px 0px -40px 0px" });

document.querySelectorAll(".reveal").forEach((el) => revealObserver.observe(el));

// The charts need their own trigger: they're canvas, so they can't just fade in.
new IntersectionObserver((entries, obs) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    animateCharts();
    obs.disconnect();
  }
}, { threshold: 0.25 }).observe($("training"));

window.addEventListener("scroll", () => {
  const max = document.body.scrollHeight - window.innerHeight;
  const pct = max > 0 ? (window.scrollY / max) * 100 : 0;
  $("scroll-progress").style.width = `${pct}%`;
  $("nav").classList.toggle("visible", window.scrollY > 260);
}, { passive: true });

// Feed the cursor position to each card's spotlight gradient.
document.querySelectorAll(".card").forEach((card) => {
  card.addEventListener("mousemove", (e) => {
    const r = card.getBoundingClientRect();
    card.style.setProperty("--mx", `${e.clientX - r.left}px`);
    card.style.setProperty("--my", `${e.clientY - r.top}px`);
  });
});

/* ---------------- chat ---------------- */

const history = [];
let streaming = false;

function addMessage(role, text = "") {
  $("empty-chat")?.remove();
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  $("chat-log").appendChild(el);
  $("chat-log").scrollTop = $("chat-log").scrollHeight;
  return el;
}

$("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (streaming) return;

  const text = $("chat-text").value.trim();
  if (!text) return;

  $("chat-text").value = "";
  addMessage("user", text);
  history.push({ role: "user", content: text });

  streaming = true;
  $("chat-send").disabled = true;
  $("chat-meta").textContent = "thinking… (each token is a full forward pass)";

  const bubble = addMessage("assistant");

  // Bouncing dots until the first token lands - the first forward pass can take
  // a while, and an empty bubble looks broken.
  const thinking = document.createElement("span");
  thinking.className = "thinking";
  thinking.innerHTML = "<span></span><span></span><span></span>";
  bubble.appendChild(thinking);

  const cursor = document.createElement("span");
  cursor.className = "cursor";

  let reply = "";
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: history }),
    });

    // SSE over a POST body, so read the stream directly rather than using
    // EventSource (which is GET-only).
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split("\n\n");
      buffer = chunks.pop();

      for (const chunk of chunks) {
        if (!chunk.startsWith("data: ")) continue;
        const evt = JSON.parse(chunk.slice(6));

        if (evt.error) {
          bubble.remove();
          addMessage("error", evt.error);
          $("chat-meta").textContent = "";
          return;
        }
        if (evt.token) {
          thinking.remove();
          reply += evt.token;
          bubble.textContent = reply;
          bubble.appendChild(cursor);
          $("chat-log").scrollTop = $("chat-log").scrollHeight;
        }
        if (evt.done) {
          $("chat-meta").textContent =
            `${evt.tokens} tokens in ${evt.elapsed.toFixed(1)}s · ` +
            `${evt.tokens_per_sec.toFixed(1)} tok/s · ${evt.tokens} forward passes`;
        }
      }
    }

    history.push({ role: "assistant", content: reply });
    if (!reply.trim()) bubble.textContent = "(empty completion)";
  } catch (err) {
    bubble.remove();
    addMessage("error", `request failed: ${err.message}`);
  } finally {
    thinking.remove();
    cursor.remove();
    streaming = false;
    $("chat-send").disabled = false;
    $("chat-text").focus();
  }
});

/* ---------------- boot ---------------- */

loadModel();
loadMetrics();
