const $ = (id) => document.getElementById(id);

/* ---------------- i18n ---------------- */

const I18N = {
  de: {
    nav: { chat: "chat", architecture: "wie es denkt", scratch: "von grund auf", personality: "persönlichkeit", evolution: "entwicklung", training: "training" },
    hero: {
      badge: "keine Abkürzungen: nichts geliehen, nichts vorgefertigt",
      acronym: "K.E.V.I.N.: Keep Emitting Vaguely Incoherent Nonsense",
      tagline: "Eine KI, die Text liest und vorhersagt, was als Nächstes kommt, Wort für Wort. Sie basiert auf "
        + "keiner bestehenden KI-Bibliothek: Jeder Teil wurde von Grund auf selbst geschrieben und dann ein "
        + "zweites Mal neu gebaut, um parallel auf einer Grafikkarte zu laufen und so viel schneller zu denken.",
    },
    stat: { params: "Parameter", layers: "Schichten", heads: "Attention-Heads", dmodel: "Netzwerkbreite", ctx: "Kontextfenster", step: "Trainingsschritte" },
    chat: {
      title: "Schreib mit ihm weiter",
      lede: "Kein Chat, sondern reine Textfortsetzung, und nur auf Englisch.",
      chatLede: "Ein echtes Hin und Her: KEVIN wurde zusätzlich auf Konversationen trainiert, "
        + "merkt sich also, was gerade gesagt wurde, und antwortet, statt nur den Satz fortzusetzen. "
        + "Auch nur auf Englisch.",
      modeAutocomplete: "Fortsetzung",
      modeChat: "Chat",
      empty: "Fang an zu schreiben, KEVIN führt es fort.",
      chatEmpty: "Sag KEVIN etwas.",
      placeholder: "Schreib etwas, das KEVIN fortsetzen soll…",
      chatPlaceholder: "Frag KEVIN etwas…",
      send: "Fortsetzen",
      chatSend: "Senden",
      continueReply: "Weiterschreiben",
      clear: "Neu anfangen",
      ttsToggle: "Antworten vorlesen",
      disabledPlaceholder: "Kein Modell geladen, Fortsetzung deaktiviert",
      disabledEmpty: "Lade ein Modell, um weiterzuschreiben.",
      chatDisabledPlaceholder: "Kein Chat-Modell geladen",
      chatDisabledEmpty: "Für den Chat-Modus muss ein feingetuntes Modell geladen werden.",
      thinking: "denkt nach … (ein Wort nach dem anderen)",
      done: (tokens, elapsed, tps) => `${tokens} Wörter in ${elapsed}s · ${tps} Wörter/s, jedes einzeln neu durchdacht`,
      requestFailed: (msg) => `Etwas ist schiefgelaufen: ${msg}`,
      emptyCompletion: "(keine Fortsetzung erhalten)",
    },
    arch: {
      title: "Wie es denkt",
      lede: "Ein Transformer vergleicht jedes Wort eines Satzes mit jedem anderen und entscheidet, welche davon "
        + "am wichtigsten sind, um das nächste Wort vorherzusagen. Das passiert immer wieder, durch viele "
        + "übereinandergestapelte Schichten, die die Vorhersage jedes Mal ein Stück schärfer machen. Jede Zahl "
        + "unten stammt live aus dem trainierten Modell, nicht zur Show erfunden.",
      tokenIds: "Eingabewörter",
      embTitle: "Embedding",
      embSub: 'verwandelt jedes Wort in Zahlen und ergänzt ein Gefühl dafür, <em>wo</em> es im Satz steht',
      blockLabel: "× übereinandergestapelte Denkschichten",
      rmsnorm: "RMSNorm",
      attnTitle: "Multi-Head Attention",
      attnNote: "blickt über alles bisher Gesagte zurück und gewichtet jedes frühere Wort gleichzeitig",
      residual: "＋ trägt das Ursprüngliche weiter",
      mlpTitle: "SwiGLU-MLP",
      mlpNote: 'eine kleine interne Entscheidung darüber, welche Ideen behalten und welche verworfen werden, allein durchs Training geformt',
      finalNorm: "letzte Prüfung",
      outTitle: "Ausgabeprojektion",
      outNote: "eine eigene, unabhängige Sicht auf den Wortschatz, getrennt davon, wie Wörter eingelesen werden",
      logits: "rohe Vorhersagen",
    },
    scratch: {
      title: "Von Grund auf gebaut",
      card1: {
        tag: "die Mathematik",
        h3: "Jedes bisschen Lernen von Hand hergeleitet",
        p1: "Die Mathematik, mit der das Modell aus seinen Fehlern lernt (Aufmerksamkeit, interne Prüfungen, "
          + "Vorhersagen) wurde auf Papier hergeleitet und direkt in Code übersetzt.",
        p2: "Sie ist so gestaltet, dass tausende Beispiele gleichzeitig verarbeitet werden können, statt "
          + "einzeln.",
      },
      card2: {
        tag: "die GPU",
        h3: "Neu gebaut, um parallel auf einer Grafikkarte zu laufen",
        p1: "Ein gewöhnlicher Prozessor arbeitet Aufgaben nacheinander ab. Eine Grafikkarte hat tausende "
          + "kleiner Kerne, die gleichzeitig arbeiten können. Deshalb wurde das Training so umgeschrieben, "
          + "dass seine Arbeit auf all diese Kerne verteilt wird. Das macht das Training deutlich schneller, "
          + "ohne etwas daran zu ändern, was das Modell lernt.",
      },
      card3: {
        tag: "der Tokenizer",
        h3: "Eine eigene Art, Text zu lesen",
        p1: "Bevor das Modell etwas lesen kann, muss Text in Stücke zerlegt werden, die es versteht. Auch "
          + "dieser Teil wurde selbst gebaut.",
        p2: "So bleibt ihm kein Wort, Emoji oder Symbol in irgendeiner Sprache je ein Rätsel.",
      },
    },
    pers: {
      title: "Kevins Persönlichkeit",
      lede: "Man gibt ihm den Anfang eines Satzes, und es macht einfach weiter. Was dabei "
        + "herauskommt, hat es sich komplett selbst aus dem Training angeeignet. Hier ist eine "
        + "Handvoll immer gleicher Satzanfänge, und was es tatsächlich damit macht.",
    },
    evo: {
      title: "Zusehen, wie es sprechen lernt",
      lede: 'Dieselbe Frage, demselben Modell gestellt, aber jeweils zu einem anderen Zeitpunkt eingefroren, '
        + 'während es noch lernte. <span id="evo-prompt-count">—</span> Momentaufnahmen, jedes Mal mit '
        + 'denselben Einstellungen. Hier vervollständigt das rohe, noch lernende Modell einen Gedanken, '
        + 'nicht der ausgefeilte Assistent, mit dem du oben gesprochen hast.',
      promptLabel: "Prompt",
      empty: (cmd) => `Noch keine Zwischenstände zum Ansehen.<br>`
        + `Führe <code>${cmd}</code> aus, um sie aus den gespeicherten Zwischenständen eines Trainingslaufs `
        + `zu erzeugen.`,
      step: "Schritt",
      loss: "Fehler",
      ppl: "Unsicherheit",
    },
    training: {
      title: "Training",
      lede: 'Ein unmittelbarer Blick darauf, wie sich das Modell im Laufe der Zeit tatsächlich verbessert '
        + 'hat, direkt aus dem Trainingslauf selbst, nichts nachträglich inszeniert oder geglättet.',
      chartLoss: "Fehler im Zeitverlauf",
      chartPpl: "Unsicherheit",
      logScale: "(vergrößerte Darstellung)",
      empty: (path) => `Noch keine Trainingszahlen vorhanden.<br>`
        + `Sie werden während des Trainings in <code>${path}</code> geschrieben, starte einen Trainingslauf, `
        + `dann erscheinen die Kurven hier.`,
      summary: { steps: "Trainingsschritte", finalLoss: "Fehler am Ende", bestLoss: "wenigste erreichte Fehler", finalPpl: "Unsicherheit am Ende", tokPerSec: "Wörter/Sek. beim Training", wallClock: "gesamte Trainingsdauer" },
    },
    loadState: {
      ok: (params, step, tokenizer, engine) =>
        `einsatzbereit, ${params} Parameter, ${step} Trainingsschritte durchlaufen, liest mit seinem ${tokenizer}-Wortschatz, denkt auf der ${engine}`,
      warnTitle: "Noch kein trainiertes Modell geladen.",
      warnBody: (ckptPath) => `Der Rest der Seite funktioniert trotzdem und zeigt, wie das Modell aufgebaut ist. `
        + `Damit der Chat lebendig wird, muss ein trainiertes Modell unter <code>${ckptPath}</code> abgelegt `
        + `und der Server neu gestartet werden (oder gestartet mit <code>python serve.py --ckpt pfad/zur.ckpt</code>).`,
      gpu: "Grafikkarte",
      cpu: "normalem Prozessor",
    },
    footer: {
      engineGpu: "läuft auf einer Grafikkarte",
      engineCpu: "läuft auf einem normalen Prozessor",
      loaded: (ckpt, tokenizer, engine) => `${ckpt} · ${tokenizer} · ${engine}`,
      noCkpt: (path) => `kein trainiertes Modell geladen · erwartet unter ${path}`,
    },
  },
  en: {
    nav: { chat: "chat", architecture: "how it thinks", scratch: "from scratch", personality: "personality", evolution: "evolution", training: "training" },
    hero: {
      badge: "no shortcuts: nothing borrowed, nothing pre-built",
      acronym: "K.E.V.I.N.: Keep Emitting Vaguely Incoherent Nonsense",
      tagline: "An AI that reads text and predicts what comes next, one word at a time. It isn't built on any "
        + "existing AI library: every part was written from scratch, then rebuilt a second time to run in "
        + "parallel on a graphics card, so it thinks far faster.",
    },
    stat: { params: "parameters", layers: "layers", heads: "attention heads", dmodel: "network width", ctx: "context window", step: "training steps" },
    chat: {
      title: "Keep writing with it",
      lede: "Not a chat, just raw text autocomplete, and English only.",
      chatLede: "A real back-and-forth: KEVIN was further finetuned on conversations, so it "
        + "remembers what you just said and answers instead of just continuing your sentence. "
        + "English only, still.",
      modeAutocomplete: "autocomplete",
      modeChat: "chat",
      empty: "Start typing and KEVIN will continue it.",
      chatEmpty: "Say something to KEVIN.",
      placeholder: "Type something for KEVIN to continue…",
      chatPlaceholder: "Ask KEVIN something…",
      send: "Continue",
      chatSend: "Send",
      continueReply: "Continue",
      clear: "Start over",
      ttsToggle: "read replies aloud",
      disabledPlaceholder: "No model loaded, continuation disabled",
      disabledEmpty: "Load a model to keep writing.",
      chatDisabledPlaceholder: "No chat model loaded",
      chatDisabledEmpty: "Load a finetuned model to enable chat mode.",
      thinking: "thinking… (working it out one word at a time)",
      done: (tokens, elapsed, tps) => `${tokens} words in ${elapsed}s · ${tps} words/s, each one freshly thought through`,
      requestFailed: (msg) => `something went wrong: ${msg}`,
      emptyCompletion: "(no continuation came back)",
    },
    arch: {
      title: "How it thinks",
      lede: "A transformer compares every word in a sentence to every other word, deciding which ones matter "
        + "most for guessing what comes next. It does this again and again through many stacked layers, each "
        + "one sharpening the guess a little further. Every number below is read live from the trained model, "
        + "not made up for show.",
      tokenIds: "input words",
      embTitle: "Embedding",
      embSub: 'turns each word into numbers, and adds a sense of <em>where</em> it sits in the sentence',
      blockLabel: "× layers of reasoning, stacked",
      rmsnorm: "RMSNorm",
      attnTitle: "Multi-Head Attention",
      attnNote: "looks back over everything said so far, weighing every earlier word at once",
      residual: "＋ carries the original signal forward",
      mlpTitle: "SwiGLU MLP",
      mlpNote: 'a small internal decision about which ideas to keep and which to drop, shaped entirely by training',
      finalNorm: "one last check",
      outTitle: "Output projection",
      outNote: "its own independent read of the vocabulary, separate from how it reads words in",
      logits: "raw predictions",
    },
    scratch: {
      title: "Built from scratch",
      card1: {
        tag: "the math",
        h3: "Every bit of learning, worked out by hand",
        p1: "The math the model uses to learn from its mistakes (attention, internal checks, predictions) "
          + "was worked out on paper and turned directly into code.",
        p2: "It's shaped so thousands of examples can be worked through at once instead of one at a time. "
          + "Training can also be told exactly which parts of a conversation to learn from, which is what "
          + "makes it possible to teach the model to hold a conversation, rather than just ramble.",
      },
      card2: {
        tag: "the gpu",
        h3: "Rebuilt to run in parallel on a graphics card",
        p1: "A regular processor works through problems one at a time. A graphics card has thousands of "
          + "small cores that can all work at once, so training was rewritten to split its work across all "
          + "of them. That's what makes training dramatically faster, without changing anything about what "
          + "the model actually learns.",
      },
      card3: {
        tag: "the tokenizer",
        h3: "Its own way of reading text",
        p1: "Before it can read anything, text has to be broken into pieces it understands. That splitter "
          + "was also built from scratch.",
        p2: "So no word, emoji, or symbol in any language is ever a mystery to it: everything can be broken "
          + "down to raw bytes as a last resort.",
      },
    },
    pers: {
      title: "Kevin's personality",
      lede: "Feed it the start of a sentence and it just keeps going. Whatever comes out is "
        + "entirely what the model picked up from training. Here's the same handful of openers, "
        + "and what it actually does with them.",
    },
    evo: {
      title: "Watching it learn to talk",
      lede: 'The exact same question, asked to the exact same model, but frozen at different moments while '
        + 'it was still learning. <span id="evo-prompt-count">—</span> snapshots, same settings every time. '
        + 'This is the raw, still-learning model completing a thought, not the polished assistant you '
        + 'chatted with above.',
      promptLabel: "prompt",
      empty: (cmd) => `No snapshots to look at yet.<br>`
        + `Run <code>${cmd}</code> to generate them from a training run's saved checkpoints.`,
      step: "step",
      loss: "mistakes",
      ppl: "uncertainty",
    },
    training: {
      title: "Training",
      lede: 'A live look at how the model actually improved over time, taken straight from the training '
        + 'run itself, nothing staged or cleaned up after the fact.',
      chartLoss: "Mistakes over time",
      chartPpl: "Uncertainty",
      logScale: "(zoomed in)",
      empty: (path) => `No training numbers yet.<br>`
        + `They get written to <code>${path}</code> as training runs, start a training run and the curves `
        + `will appear here.`,
      summary: { steps: "training steps", finalLoss: "mistakes at the end", bestLoss: "fewest mistakes reached", finalPpl: "uncertainty at the end", tokPerSec: "words/sec while training", wallClock: "total time spent training" },
    },
    loadState: {
      ok: (params, step, tokenizer, engine) =>
        `up and running, ${params} parameters, ${step} training steps in, reading with its ${tokenizer} vocabulary, thinking on the ${engine}`,
      warnTitle: "No trained model loaded yet.",
      warnBody: (ckptPath) => `The rest of the page still works and shows how the model is built. To bring `
        + `the chat to life, a trained model needs to be placed at <code>${ckptPath}</code> and the server `
        + `restarted (or launched with <code>python serve.py --ckpt path/to/your.ckpt</code>).`,
      gpu: "graphics card",
      cpu: "regular processor",
    },
    footer: {
      engineGpu: "running on a graphics card",
      engineCpu: "running on a regular processor",
      loaded: (ckpt, tokenizer, engine) => `${ckpt} · ${tokenizer} · ${engine}`,
      noCkpt: (path) => `no trained model loaded · expected at ${path}`,
    },
  },
};

let LANG = localStorage.getItem("kevin-lang") || "de";

/** Looks up a dotted path in the current language's dictionary, e.g. t("chat.title"). */
function t(path) {
  const parts = path.split(".");
  let node = I18N[LANG];
  for (const p of parts) node = node?.[p];
  return node;
}

function applyStaticTranslations() {
  document.documentElement.lang = LANG;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const v = t(el.getAttribute("data-i18n"));
    if (v != null) el.textContent = v;
  });
  document.querySelectorAll("[data-i18n-html]").forEach((el) => {
    const v = t(el.getAttribute("data-i18n-html"));
    if (v != null) el.innerHTML = v;
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    const v = t(el.getAttribute("data-i18n-placeholder"));
    if (v != null) el.placeholder = v;
  });
  const toggle = $("lang-toggle");
  if (toggle) toggle.textContent = LANG === "de" ? "EN" : "DE";
}

/** Re-applies language: static markup, plus dynamic sections rendered from cached fetch data. */
function setLanguage(lang) {
  LANG = lang;
  localStorage.setItem("kevin-lang", lang);
  applyStaticTranslations();
  renderModelInfo();
  renderMetrics();
  renderEvolution();
  renderPersonality();
  applyModeAvailability();
}

const fmtInt = (n) => Math.round(n).toLocaleString(LANG === "de" ? "de-DE" : "en-US");

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
  renderModelInfo();
}

function renderModelInfo() {
  if (!MODEL) return;
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

    const engine = MODEL.engine === "cuda" ? t("loadState.gpu") : t("loadState.cpu");
    $("load-state").innerHTML =
      `<span class="ok">●</span> ` +
      t("loadState.ok")(fmtInt(MODEL.params), fmtInt(MODEL.step), MODEL.tokenizer, engine);
  } else {
    // No weights: the showcase still stands on its own, but be explicit that
    // the chat can't run rather than quietly serving noise.
    $("load-state").innerHTML =
      `<div class="warn"><b>${t("loadState.warnTitle")}</b><br>` +
      `${escapeHtml(MODEL.error || "")}<br><br>` +
      t("loadState.warnBody")(escapeHtml(MODEL.ckpt_path)) + `</div>`;
  }

  $("mode-chat").disabled = !MODEL.chat?.model_loaded;
  applyModeAvailability();

  const engineLabel = MODEL.engine === "cuda" ? t("footer.engineGpu") : t("footer.engineCpu");
  $("footer-meta").textContent = MODEL.model_loaded
    ? t("footer.loaded")(MODEL.ckpt_path, MODEL.tokenizer, engineLabel)
    : t("footer.noCkpt")(MODEL.ckpt_path);
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
  renderMetrics();
}

function renderMetrics() {
  if (!METRICS) return;

  if (!METRICS.available) {
    document.querySelector(".charts").hidden = true;
    const empty = $("metrics-empty");
    empty.hidden = false;
    empty.innerHTML = t("training.empty")(escapeHtml(METRICS.path));
    return;
  }

  document.querySelector(".charts").hidden = false;
  $("metrics-empty").hidden = true;

  const s = METRICS.summary;
  const sm = t("training.summary");
  $("metrics-summary").innerHTML = `
    <div><b>${fmtInt(s.steps)}</b>${sm.steps}</div>
    <div><b>${s.final_loss.toFixed(4)}</b>${sm.finalLoss}</div>
    <div><b>${s.best_loss.toFixed(4)}</b>${sm.bestLoss}</div>
    <div><b>${s.final_perplexity.toFixed(1)}</b>${sm.finalPpl}</div>
    <div><b>${fmtInt(s.mean_tokens_per_sec)}</b>${sm.tokPerSec}</div>
    <div><b>${(s.elapsed_sec / 60).toFixed(1)}m</b>${sm.wallClock}</div>`;

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

/* ---------------- personality ----------------
 * A handful of fixed openers, run through raw autocomplete (no chat template),
 * so whatever comes back is unfiltered model personality. Completions below are
 * copied verbatim from real runs against /api/chat with raw: true - typos and
 * non-sequiturs included, since that's the actual point of the section. */

const PERSONALITY = [
  { prompt: "Donald Trump is", tag: "unhinged",
    completion: "lost through the development of many mental health disorders, such as depression." },
  { prompt: "The best recipe is", tag: "non sequitur",
    completion: "a good old man who has been in the house for many years. He is cautious, all the time." },
  { prompt: "I'm an AI. My biggest fear is", tag: "existential",
    completion: "that I may be a person who does not think of me as an AI." },
  { prompt: "In my free time, I like to", tag: "circular logic",
    completion: "write my own stories and try to read them and learn how to read them." },
  { prompt: "Honestly, the worst thing about Germany is", tag: "backhanded",
    completion: "that they could take a long period of time to recover from their own mistakes." },
  { prompt: "Berlin is", tag: "earnest",
    completion: "a city where you can experience life as an artist." },
];

function renderPersonality() {
  const grid = $("pers-grid");
  if (!grid) return;
  grid.innerHTML = "";

  PERSONALITY.forEach((p, i) => {
    const card = document.createElement("article");
    card.className = "pers-card reveal";
    card.style.setProperty("--i", i);

    card.innerHTML = `
      <div class="pers-tag">${escapeHtml(p.tag)}</div>
      <p class="pers-prompt">${escapeHtml(p.prompt)}</p>
      <p class="pers-completion">${escapeHtml(p.completion)}</p>
    `;

    grid.appendChild(card);
    revealObserver.observe(card);
  });
}

/* ---------------- evolution timeline ---------------- */

let EVOLUTION = null;

async function loadEvolution() {
  try {
    EVOLUTION = await (await fetch("/static/data/evolution.json")).json();
  } catch {
    EVOLUTION = null;
  }
  renderEvolution();
}

function renderEvolution() {
  const timeline = $("evo-timeline");
  const data = EVOLUTION;

  if (!data || !data.samples || !data.samples.length) {
    timeline.hidden = true;
    const empty = $("evo-empty");
    empty.hidden = false;
    empty.innerHTML = t("evo.empty")("python tools_gen_evolution.py");
    return;
  }

  timeline.hidden = false;
  $("evo-empty").hidden = true;
  timeline.innerHTML = "";

  $("evo-prompt").textContent = `"${data.prompt}"`;
  $("evo-prompt-count").textContent = data.samples.length;

  data.samples.forEach((s, i) => {
    const card = document.createElement("div");
    card.className = "evo-card reveal";
    card.style.setProperty("--i", i);

    const meta = [`${t("evo.step")} ${fmtInt(s.step)}`];
    if (s.loss != null) meta.push(`${t("evo.loss")} ${s.loss.toFixed(3)}`);
    if (s.perplexity != null) meta.push(`${t("evo.ppl")} ${s.perplexity.toFixed(1)}`);

    const stepEl = document.createElement("div");
    stepEl.className = "evo-step";
    stepEl.textContent = fmtParams(s.step).replace(".0", "");

    const body = document.createElement("div");
    body.className = "evo-body";
    body.innerHTML = `<div class="evo-meta">${meta.map(escapeHtml).join(" · ")}</div>`;

    const text = document.createElement("p");
    text.className = "evo-text";
    text.innerHTML =
      `<span class="evo-prompt-echo">${escapeHtml(data.prompt)}</span>` +
      escapeHtml(s.text);
    body.appendChild(text);

    card.appendChild(stepEl);
    card.appendChild(body);
    timeline.appendChild(card);
    revealObserver.observe(card);
  });
}

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

/* ---------------- autocomplete / chat ----------------
 * Two ways to talk to KEVIN, toggled by the mode-switch above the chat log:
 *
 * - "autocomplete" hits the base model. There's no <|user|>/<|assistant|>
 *   template underneath: whatever you've typed plus everything the model has
 *   already continued is one growing block of raw text, and each submit just
 *   asks the model to keep going from the end of it (raw: true).
 * - "chat" hits the separate SFT-finetuned checkpoint (STATE["chat_model"] on
 *   the server) through the normal <|user|>/<|assistant|> template (raw:
 *   false), as real back-and-forth turns rather than one continuous blob. */

let MODE = "autocomplete"; // "autocomplete" | "chat"

// The server clamps this to max(16, context_window / 2) (128 tokens on every
// checkpoint this project has trained, see serve.py's _max_new_tokens_cap).
// Kept well under that cap: a real chat reply from this small, 2k-example
// SFT model is a sentence or two - letting it run to 128 tokens just gives a
// turn that fails to stop more room to ramble before the budget runs out.
const MAX_NEW_TOKENS = 64;

let docText = "";
let docBubble = null;
let chatMessages = [];
let lastAssistantBubble = null; // chat mode only - what "Continue" appends to
let streaming = false;
let streamAbort = null;

function addMessage(role, text = "") {
  $("empty-chat")?.remove();
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  $("chat-log").appendChild(el);
  $("chat-log").scrollTop = $("chat-log").scrollHeight;
  return el;
}

/** The base-model or chat-model slice of /api/model, matching the active mode. */
function modeInfo() {
  if (!MODEL) return null;
  return MODE === "chat" ? MODEL.chat : MODEL;
}

/** Enables/disables the text input and the send/continue buttons based on
 * whether the active mode's model is loaded, whether a generation is already
 * streaming, and (for "Continue") whether there's a finished assistant reply
 * to extend. Doesn't touch labels/placeholders - see applyModeAvailability. */
function updateActionButtons() {
  const loaded = !!modeInfo()?.model_loaded;
  const canAct = loaded && !streaming;
  $("chat-text").disabled = !canAct;
  $("chat-send").disabled = !canAct;
  const hasReply = MODE === "chat" && chatMessages.length > 0
    && chatMessages[chatMessages.length - 1].role === "assistant";
  $("chat-continue").disabled = !canAct || !hasReply;
}

function applyModeAvailability() {
  const loaded = !!modeInfo()?.model_loaded;
  $("chat-continue").hidden = MODE !== "chat";
  $("chat-send").textContent = t(MODE === "chat" ? "chat.chatSend" : "chat.send");
  $("chat-continue").textContent = t("chat.continueReply");
  $("chat-text").placeholder = t(loaded
    ? (MODE === "chat" ? "chat.chatPlaceholder" : "chat.placeholder")
    : (MODE === "chat" ? "chat.chatDisabledPlaceholder" : "chat.disabledPlaceholder"));
  const emptyEl = $("empty-chat");
  if (emptyEl) {
    emptyEl.textContent = t(loaded
      ? (MODE === "chat" ? "chat.chatEmpty" : "chat.empty")
      : (MODE === "chat" ? "chat.chatDisabledEmpty" : "chat.disabledEmpty"));
  }
  const lede = $("chat-lede");
  if (lede) lede.textContent = t(MODE === "chat" ? "chat.chatLede" : "chat.lede");
  updateActionButtons();
}

function switchMode(mode) {
  if (mode === MODE || streaming) return;
  MODE = mode;
  $("mode-autocomplete").classList.toggle("active", mode === "autocomplete");
  $("mode-autocomplete").setAttribute("aria-selected", String(mode === "autocomplete"));
  $("mode-chat").classList.toggle("active", mode === "chat");
  $("mode-chat").setAttribute("aria-selected", String(mode === "chat"));
  resetChatSession();
}

$("mode-autocomplete").addEventListener("click", () => switchMode("autocomplete"));
$("mode-chat").addEventListener("click", () => switchMode("chat"));

// A same-tab reload should always start KEVIN from a blank page. Plain JS
// state (docText etc.) already resets on a real navigation, but a fast
// refresh can be served from the browser's back/forward cache, which
// freezes and restores the whole JS heap instead of re-running this script -
// so docText, the chat log DOM, and any in-flight stream from before the
// refresh would otherwise survive into the "new" session. Cancel any
// in-flight stream before the page is frozen, and rebuild state from
// scratch whenever the page is (re)shown, bfcache or not.
function resetChatSession() {
  streamAbort?.abort();
  streamAbort = null;
  streaming = false;
  if (TTS_SUPPORTED) speechSynthesis.cancel();
  docText = "";
  docBubble = null;
  chatMessages = [];
  lastAssistantBubble = null;
  $("chat-log").innerHTML = '<div class="empty-chat" id="empty-chat"></div>';
  $("chat-text").value = "";
  $("chat-meta").textContent = "";
  applyStaticTranslations();
  applyModeAvailability();
}

// Unconditional, not just on bfcache-restore (e.persisted): a fresh
// navigation already starts every var empty, so this is a no-op then, but it
// guarantees any restart - bfcache, tab-restore-after-crash, whatever the
// browser does under the hood - always lands on a genuinely blank session.
window.addEventListener("pageshow", () => resetChatSession());
window.addEventListener("pagehide", () => streamAbort?.abort());

$("chat-clear").addEventListener("click", () => resetChatSession());

/** Streams one /api/chat completion, feeding text deltas to onToken and the
 * final {tokens, elapsed, tokens_per_sec} to onDone. Shared by both modes -
 * they only differ in what goes into `body` and what onToken/onDone do with
 * the result. */
async function streamCompletion(body, { onToken, onDone }) {
  streamAbort = new AbortController();
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: streamAbort.signal,
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
        addMessage("error", evt.error);
        $("chat-meta").textContent = "";
        return;
      }
      if (evt.token) onToken(evt.token);
      if (evt.done) onDone(evt);
    }
  }
}

/** Drives one full generation into `replyBubble`, starting from
 * `initialText` (empty for a fresh reply, the reply-so-far for "Continue").
 * `onFinish(finalText, newText)` is called once the stream completes, so the
 * caller can file the result away (push a new assistant message, or extend
 * the last one) - `newText` is just the slice generated this call, handy for
 * TTS so "Continue" doesn't re-read what was already spoken. Shared by the
 * send and continue handlers below - they only differ in what `body` asks
 * the server for and what onFinish does. */
async function runGeneration(body, replyBubble, initialText, onFinish) {
  let replyText = initialText;

  streaming = true;
  updateActionButtons();
  $("chat-meta").textContent = t("chat.thinking");

  // Bouncing dots until the first token lands - the first forward pass can take
  // a while, and an empty bubble looks broken.
  const thinking = document.createElement("span");
  thinking.className = "thinking";
  thinking.innerHTML = "<span></span><span></span><span></span>";
  replyBubble.appendChild(thinking);

  const cursor = document.createElement("span");
  cursor.className = "cursor";

  try {
    await streamCompletion(body, {
      onToken: (token) => {
        thinking.remove();
        replyText += token;
        if (MODE === "autocomplete") docText = replyText;
        replyBubble.textContent = replyText;
        replyBubble.appendChild(cursor);
        $("chat-log").scrollTop = $("chat-log").scrollHeight;
      },
      onDone: (evt) => {
        onFinish(replyText, replyText.slice(initialText.length));
        $("chat-meta").textContent =
          t("chat.done")(evt.tokens, evt.elapsed.toFixed(1), evt.tokens_per_sec.toFixed(1));
      },
    });
  } catch (err) {
    if (err.name !== "AbortError") addMessage("error", t("chat.requestFailed")(err.message));
  } finally {
    thinking.remove();
    cursor.remove();
    streaming = false;
    streamAbort = null;
    updateActionButtons();
    $("chat-text").focus();
  }
}

$("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (streaming) return;

  const typed = $("chat-text").value;

  if (MODE === "chat") {
    if (!typed) return;
    $("chat-text").value = "";
    chatMessages.push({ role: "user", content: typed });
    addMessage("user", typed);

    lastAssistantBubble = addMessage("assistant");
    const body = { messages: chatMessages, mode: "chat", raw: false, max_new_tokens: MAX_NEW_TOKENS };
    await runGeneration(body, lastAssistantBubble, "", (finalText, newText) => {
      chatMessages.push({ role: "assistant", content: finalText });
      speak(newText);
    });
  } else {
    if (!typed && !docText) return;
    if (typed) {
      docText += docText && !/\s$/.test(docText) && !/^\s/.test(typed) ? " " + typed : typed;
    }
    $("chat-text").value = "";
    if (!docBubble) docBubble = addMessage("doc");
    docBubble.textContent = docText;

    const body = { messages: [{ role: "user", content: docText }], mode: "autocomplete", raw: true, max_new_tokens: MAX_NEW_TOKENS };
    await runGeneration(body, docBubble, docText, (finalText, newText) => speak(newText));
  }
});

// "Continue" only exists in chat mode: it asks the model to keep going on
// its last reply (via continue_reply on the server) instead of opening a
// fresh turn - handy when the 128-token cap cuts a reply off mid-thought.
$("chat-continue").addEventListener("click", async () => {
  if (streaming || MODE !== "chat") return;
  if (!chatMessages.length || chatMessages[chatMessages.length - 1].role !== "assistant") return;
  if (!lastAssistantBubble) return;

  const idx = chatMessages.length - 1;
  const body = {
    messages: chatMessages, mode: "chat", raw: false,
    continue_reply: true, max_new_tokens: MAX_NEW_TOKENS,
  };
  await runGeneration(body, lastAssistantBubble, chatMessages[idx].content, (finalText, newText) => {
    chatMessages[idx].content = finalText;
    speak(newText);
  });
});

/* ---------------- text-to-speech ----------------
 * Reads finished KEVIN replies aloud via the browser's built-in speech
 * synthesis - no server involvement, so it works the same whether the
 * chat hits the base model or the finetuned one. Off by default and
 * remembered across visits; hidden entirely on browsers without the API. */

const TTS_SUPPORTED = "speechSynthesis" in window;
let ttsEnabled = TTS_SUPPORTED && localStorage.getItem("kevin-tts") === "1";
let ttsVoice = null;

function pickTtsVoice() {
  const voices = speechSynthesis.getVoices();
  ttsVoice = voices.find((v) => v.lang === "en-US")
    || voices.find((v) => v.lang?.startsWith("en"))
    || voices[0]
    || null;
}

if (TTS_SUPPORTED) {
  pickTtsVoice();
  speechSynthesis.addEventListener("voiceschanged", pickTtsVoice);

  const ttsToggle = $("tts-toggle");
  ttsToggle.hidden = false;
  ttsToggle.setAttribute("aria-pressed", String(ttsEnabled));
  ttsToggle.addEventListener("click", () => {
    ttsEnabled = !ttsEnabled;
    localStorage.setItem("kevin-tts", ttsEnabled ? "1" : "0");
    ttsToggle.setAttribute("aria-pressed", String(ttsEnabled));
    if (!ttsEnabled) speechSynthesis.cancel();
  });
}

/** Speaks `text` (a finished reply, or just the newly-generated slice of
 * one) if the toggle is on. No-op when TTS is unsupported/disabled/empty. */
function speak(text) {
  if (!TTS_SUPPORTED || !ttsEnabled || !text || !text.trim()) return;
  const utter = new SpeechSynthesisUtterance(text);
  if (ttsVoice) utter.voice = ttsVoice;
  utter.lang = ttsVoice?.lang || "en-US";
  speechSynthesis.speak(utter);
}

/* ---------------- paper peek modal ---------------- */

{
  const peek = $("paper-peek");
  const modal = $("paper-modal");
  const closeBtn = $("paper-modal-close");

  const openModal = () => modal.classList.add("open");
  const closeModal = () => modal.classList.remove("open");

  peek.addEventListener("click", openModal);
  peek.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      openModal();
    }
  });
  closeBtn.addEventListener("click", closeModal);
  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeModal();
  });
}

/* ---------------- boot ---------------- */

applyStaticTranslations();
$("lang-toggle").addEventListener("click", () => setLanguage(LANG === "de" ? "en" : "de"));

renderPersonality();
loadModel();
loadMetrics();
loadEvolution();
