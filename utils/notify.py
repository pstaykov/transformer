"""Email notifications for training progress.

Sends a nicely formatted "KEVIN said: ..." email at a configurable step
interval, containing text greedily sampled from the model being trained
from a couple of fixed starting prompts, plus loss/perplexity charts
rendered from metrics.csv.

Credentials are read from a gitignored config file (default: email.env in
the repo root), a simple KEY=VALUE file with:

    EMAIL_USER=you@gmail.com
    EMAIL_PASS=xxxxxxxxxxxxxxxx   # Gmail App Password, not your login password
    EMAIL_TO=you@gmail.com,someone_else@gmail.com   # comma-separated for multiple recipients

Generate one at https://myaccount.google.com/apppasswords (requires 2FA
enabled on the Google account).
"""

import csv
import html
import io
import os
import smtplib
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

DEFAULT_PROMPTS = ["Hello, I'm Kevin and", "The"]


def load_env_file(path):
    config = {}
    if not os.path.exists(path):
        return config
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            config[key.strip()] = value.strip()
    return config


def send_email(subject, text_body, html_body, config, images=None):
    """images: dict of content_id -> png bytes, referenced in html_body as cid:content_id"""
    user = config.get("EMAIL_USER")
    password = config.get("EMAIL_PASS")
    to_addrs = [a.strip() for a in config.get("EMAIL_TO", user).split(",") if a.strip()]
    if not user or not password:
        raise RuntimeError(
            "Missing EMAIL_USER/EMAIL_PASS. Create an email.env file (see utils/notify.py "
            "docstring) or pass --email-config pointing to one."
        )

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = ", ".join(to_addrs)

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(text_body, "plain"))
    alt.attach(MIMEText(html_body, "html"))
    msg.attach(alt)

    for cid, png_bytes in (images or {}).items():
        img = MIMEImage(png_bytes, "png")
        img.add_header("Content-ID", f"<{cid}>")
        img.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
        msg.attach(img)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(user, password)
        server.sendmail(user, to_addrs, msg.as_string())


def generate(model, tokenizer, prompt, num_tokens, max_len, seed=None):
    """Greedily generate `num_tokens` tokens continuing from `prompt`."""
    ids = list(tokenizer.encode(prompt))

    for _ in range(num_tokens):
        context = ids[-max_len:]
        x = np.array([context], dtype=np.int64)
        logits = model.forward(x)
        next_id = int(np.argmax(logits[0, -1]))
        ids.append(next_id)

    return tokenizer.decode(ids)


def _read_metrics(metrics_path):
    if not metrics_path or not os.path.exists(metrics_path):
        return None
    steps, losses, ppls = [], [], []
    with open(metrics_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                steps.append(int(row["step"]))
                losses.append(float(row["loss"]))
                ppls.append(float(row["perplexity"]))
            except (KeyError, ValueError):
                continue
    if not steps:
        return None
    return steps, losses, ppls


def render_metrics_png(metrics_path):
    """Render loss + perplexity curves from metrics.csv as a single PNG. None if unavailable."""
    if not HAS_MATPLOTLIB:
        return None
    data = _read_metrics(metrics_path)
    if data is None:
        return None
    steps, losses, ppls = data

    fig, (ax_loss, ax_ppl) = plt.subplots(1, 2, figsize=(9, 3.2), dpi=140)
    for ax, ys, title, color in (
        (ax_loss, losses, "loss", "#4c6ef5"),
        (ax_ppl, ppls, "perplexity", "#f76707"),
    ):
        ax.plot(steps, ys, color=color, linewidth=1.5)
        ax.set_title(title, fontsize=11, color="#212529")
        ax.set_xlabel("step", fontsize=9, color="#495057")
        ax.tick_params(labelsize=8, colors="#495057")
        ax.grid(True, alpha=0.25)
        for spine in ax.spines.values():
            spine.set_color("#ced4da")
    fig.patch.set_facecolor("white")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _build_text_body(step, samples, stats, has_chart, note=None):
    lines = [f"KEVIN's progress report -- step {step:,}"]
    if stats:
        stat_line = " | ".join(f"{k}: {v}" for k, v in stats.items())
        lines.append(stat_line)
    lines.append("")
    if note:
        lines.append(f"note: {note}")
        lines.append("")
    for prompt, completion in samples:
        lines.append(f'KEVIN said: "{completion}"')
        lines.append(f"  (started from: {prompt!r})")
        lines.append("")
    if has_chart:
        lines.append("(loss/perplexity charts attached below in HTML clients)")
        lines.append("")
    lines.append(f"-- sent {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(lines)


def _build_html_body(step, samples, stats, chart_cid, note=None):
    note_html = ""
    if note:
        note_html = f"""
        <div style="margin:14px 0;padding:12px 16px;background:#fff9db;
                    border-left:4px solid #f59f00;border-radius:6px;
                    font-size:14px;color:#495057;font-family:-apple-system,Helvetica,Arial,sans-serif;">
          {html.escape(note)}
        </div>"""

    stat_html = ""
    if stats:
        chips = "".join(
            f'<span style="display:inline-block;background:#f1f3f5;color:#495057;'
            f'border-radius:6px;padding:4px 10px;margin:0 6px 6px 0;font-size:13px;'
            f'font-family:-apple-system,Helvetica,Arial,sans-serif;">'
            f'{html.escape(str(k))}: <b>{html.escape(str(v))}</b></span>'
            for k, v in stats.items()
        )
        stat_html = f'<div style="margin:12px 0;">{chips}</div>'

    sample_html = ""
    for prompt, completion in samples:
        sample_html += f"""
        <div style="margin:16px 0;padding:14px 18px;background:#f8f9fa;
                    border-left:4px solid #4c6ef5;border-radius:6px;">
          <div style="font-size:15px;color:#212529;font-family:-apple-system,Helvetica,Arial,sans-serif;">
            KEVIN said: <span style="font-style:italic;">&ldquo;{html.escape(completion)}&rdquo;</span>
          </div>
          <div style="margin-top:6px;font-size:12px;color:#868e96;font-family:-apple-system,Helvetica,Arial,sans-serif;">
            started from: <code>{html.escape(prompt)}</code>
          </div>
        </div>"""

    chart_html = ""
    if chart_cid:
        chart_html = f"""
        <div style="margin:20px 0 4px 0;">
          <img src="cid:{chart_cid}" alt="loss/perplexity charts"
               style="width:100%;max-width:512px;border-radius:6px;border:1px solid #e9ecef;" />
        </div>"""

    return f"""\
<html>
  <body style="margin:0;padding:0;background:#eef0f2;">
    <div style="max-width:560px;margin:24px auto;background:#ffffff;border-radius:10px;
                overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.08);
                font-family:-apple-system,Helvetica,Arial,sans-serif;">
      <div style="background:#212529;color:#ffffff;padding:18px 24px;">
        <div style="font-size:18px;font-weight:600;">KEVIN's progress report</div>
        <div style="font-size:13px;color:#adb5bd;margin-top:2px;">step {step:,}</div>
      </div>
      <div style="padding:20px 24px;">
        {stat_html}
        {note_html}
        {sample_html}
        {chart_html}
        <div style="margin-top:18px;font-size:11px;color:#adb5bd;">
          sent {html.escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}
        </div>
      </div>
    </div>
  </body>
</html>"""


def notify_progress(model, tokenizer, step, config, max_len, num_tokens=20,
                     prompts=None, stats=None, metrics_path=None, note=None):
    prompts = prompts or DEFAULT_PROMPTS
    samples = [(p, generate(model, tokenizer, p, num_tokens, max_len)) for p in prompts]

    chart_png = render_metrics_png(metrics_path)
    images = {}
    chart_cid = None
    if chart_png:
        chart_cid = "metrics_chart"
        images[chart_cid] = chart_png

    subject = f"KEVIN said: {samples[0][1][:60]}"
    text_body = _build_text_body(step, samples, stats, has_chart=bool(chart_png), note=note)
    html_body = _build_html_body(step, samples, stats, chart_cid, note=note)
    send_email(subject, text_body, html_body, config, images=images)
    return samples


def send_note(note, config, step=None, stats=None, metrics_path=None, subject=None):
    """Send a one-off email carrying just a text note (+ metrics chart if available).

    No model/tokenizer needed - meant for an outside observer (human or LLM
    watching logs) to fire off a quick status update, e.g.:

        from utils import notify
        config = notify.load_env_file("email.env")
        notify.send_note("training running clean, loss trending down, GPU temps normal", config)
    """
    if step is None:
        data = _read_metrics(metrics_path)
        step = data[0][-1] if data else 0

    chart_png = render_metrics_png(metrics_path)
    images = {}
    chart_cid = None
    if chart_png:
        chart_cid = "metrics_chart"
        images[chart_cid] = chart_png

    subject = subject or f"KEVIN update: {note[:60]}"
    text_body = _build_text_body(step, [], stats, has_chart=bool(chart_png), note=note)
    html_body = _build_html_body(step, [], stats, chart_cid, note=note)
    send_email(subject, text_body, html_body, config, images=images)
