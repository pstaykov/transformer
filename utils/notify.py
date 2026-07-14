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
    EMAIL_FROM_NAME=KEVIN         # optional, display name shown instead of the raw address

Defaults to Gmail's SMTP (smtp.gmail.com:465, implicit TLS). To use a
different provider, add:

    EMAIL_SMTP_HOST=smtp-mail.outlook.com
    EMAIL_SMTP_PORT=587
    EMAIL_SMTP_STARTTLS=true      # Outlook/most non-Gmail providers use STARTTLS on 587
                                   # instead of implicit TLS on 465

Gmail App Password: https://myaccount.google.com/apppasswords (requires 2FA).
Outlook: use your normal password, or an app password if 2FA is enabled
(https://account.live.com/proofs/AppPassword).

A brand-new mailbox has little sending reputation, so its outbound mail can
be silently held/dropped by the provider even after the SMTP send succeeds
(no bounce, no spam-folder entry - it just never arrives). Use an
established mailbox you already send/receive real mail from.
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
from email.utils import formataddr, formatdate, make_msgid

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

    display_name = config.get("EMAIL_FROM_NAME", "KEVIN")

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = formataddr((display_name, user))
    msg["To"] = ", ".join(to_addrs)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=user.rpartition("@")[2] or "localhost")
    msg["List-Unsubscribe"] = f"<mailto:{user}?subject=unsubscribe>"
    msg["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(text_body, "plain"))
    alt.attach(MIMEText(html_body, "html"))
    msg.attach(alt)

    for cid, png_bytes in (images or {}).items():
        img = MIMEImage(png_bytes, "png")
        img.add_header("Content-ID", f"<{cid}>")
        img.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
        msg.attach(img)

    host = config.get("EMAIL_SMTP_HOST", "smtp.gmail.com")
    port = int(config.get("EMAIL_SMTP_PORT", 465))
    use_starttls = config.get("EMAIL_SMTP_STARTTLS", "").strip().lower() in ("1", "true", "yes")

    if use_starttls:
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, to_addrs, msg.as_string())
    else:
        with smtplib.SMTP_SSL(host, port) as server:
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
        ax.set_yscale("log")
        ax.set_title(title, fontsize=11, color="#212529")
        ax.set_xlabel("step", fontsize=9, color="#495057")
        ax.tick_params(labelsize=8, colors="#495057")
        ax.grid(True, which="both", alpha=0.25)
        for spine in ax.spines.values():
            spine.set_color("#ced4da")
    fig.patch.set_facecolor("white")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _build_text_body(step, samples, stats, has_chart, sender=None, note=None):
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
    if sender:
        lines.append(f"To unsubscribe, reply to {sender} with subject 'unsubscribe'.")
    return "\n".join(lines)


def _build_html_body(step, samples, stats, chart_cid, sender=None, note=None):
    font = "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;"

    note_html = ""
    if note:
        note_html = f"""
        <div style="margin:16px 0;padding:13px 18px;background:#fff9db;
                    border-left:4px solid #f59f00;border-radius:8px;
                    font-size:14px;line-height:1.5;color:#495057;{font}">
          {html.escape(note)}
        </div>"""

    stat_html = ""
    if stats:
        chips = "".join(
            f'<span style="display:inline-block;background:#f1f3f5;color:#495057;'
            f'border-radius:8px;padding:5px 12px;margin:0 6px 6px 0;font-size:13px;{font}">'
            f'{html.escape(str(k))}: <b style="color:#212529;">{html.escape(str(v))}</b></span>'
            for k, v in stats.items()
        )
        stat_html = f'<div style="margin:14px 0;">{chips}</div>'

    font = "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;"
    mono = "font-family:ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,monospace;"

    sample_html = ""
    for prompt, completion in samples:
        sample_html += f"""
        <div style="margin:18px 0;border-radius:10px;overflow:hidden;
                    border:1px solid #e9ecef;box-shadow:0 1px 2px rgba(0,0,0,0.04);">
          <div style="padding:12px 18px;background:#f8f9fa;border-bottom:1px solid #e9ecef;">
            <div style="font-size:10px;font-weight:700;letter-spacing:0.08em;color:#868e96;
                        text-transform:uppercase;{font}margin-bottom:5px;">Prompt</div>
            <div style="font-size:14px;color:#495057;{mono}">{html.escape(prompt)}</div>
          </div>
          <div style="padding:14px 18px;background:#eef2ff;">
            <div style="font-size:10px;font-weight:700;letter-spacing:0.08em;color:#4c6ef5;
                        text-transform:uppercase;{font}margin-bottom:5px;">Kevin said</div>
            <div style="font-size:15px;line-height:1.5;color:#212529;font-style:italic;{font}">
              &ldquo;{html.escape(completion)}&rdquo;
            </div>
          </div>
        </div>"""

    chart_html = ""
    if chart_cid:
        chart_html = f"""
        <div style="margin:22px 0 4px 0;">
          <img src="cid:{chart_cid}" alt="loss/perplexity charts"
               style="width:100%;max-width:512px;border-radius:8px;border:1px solid #e9ecef;" />
        </div>"""

    unsubscribe_html = ""
    if sender:
        unsubscribe_html = f"""
        <div style="text-align:center;margin-top:16px;">
          <a href="mailto:{html.escape(sender)}?subject=unsubscribe"
             style="display:inline-block;font-size:11px;color:#adb5bd;text-decoration:underline;{font}">
            unsubscribe
          </a>
        </div>"""

    return f"""\
<html>
  <body style="margin:0;padding:0;background:#eef0f2;">
    <div style="max-width:580px;margin:28px auto;background:#ffffff;border-radius:14px;
                overflow:hidden;box-shadow:0 2px 10px rgba(0,0,0,0.08);{font}">
      <div style="background:linear-gradient(135deg,#212529,#343a40);padding:22px 26px;">
        <div style="font-size:20px;font-weight:700;letter-spacing:-0.01em;color:#ffffff !important;">🤖 KEVIN&rsquo;s progress report</div>
        <div style="font-size:13px;color:#adb5bd !important;margin-top:4px;">step {step:,}</div>
      </div>
      <div style="padding:22px 26px;">
        {stat_html}
        {note_html}
        {sample_html}
        {chart_html}
        <div style="margin-top:20px;padding-top:14px;border-top:1px solid #f1f3f5;
                    font-size:11px;color:#adb5bd;">
          sent {html.escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}
        </div>
        {unsubscribe_html}
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

    sender = config.get("EMAIL_USER")
    subject = f"KEVIN said: {samples[0][1][:60]}"
    text_body = _build_text_body(step, samples, stats, has_chart=bool(chart_png), sender=sender, note=note)
    html_body = _build_html_body(step, samples, stats, chart_cid, sender=sender, note=note)
    send_email(subject, text_body, html_body, config, images=images)
    return samples


def send_note(note, config, step=None, stats=None, metrics_path=None, subject=None,
              prompt=None, completion=None, samples=None):
    """Send a one-off email carrying just a text note (+ metrics chart if available).

    No model/tokenizer needed - meant for an outside observer (human or LLM
    watching logs) to fire off a quick status update, e.g.:

        from utils import notify
        config = notify.load_env_file("email.env")
        notify.send_note("training running clean, loss trending down, GPU temps normal", config)

    Pass `prompt`/`completion` together to render a single generated sample as
    the usual two-box "Prompt" / "Kevin said" layout instead of (or alongside)
    a plain note. Pass `samples` (a list of (prompt, completion) tuples) for
    multiple samples - each gets its own pair of boxes.
    """
    if step is None:
        data = _read_metrics(metrics_path)
        step = data[0][-1] if data else 0

    if samples is None:
        samples = [(prompt, completion)] if prompt is not None and completion is not None else []

    chart_png = render_metrics_png(metrics_path)
    images = {}
    chart_cid = None
    if chart_png:
        chart_cid = "metrics_chart"
        images[chart_cid] = chart_png

    sender = config.get("EMAIL_USER")
    first_completion = samples[0][1] if samples else None
    subject = subject or (f"KEVIN said: {first_completion[:60]}" if first_completion else f"KEVIN update: {note[:60]}")
    text_body = _build_text_body(step, samples, stats, has_chart=bool(chart_png), sender=sender, note=note)
    html_body = _build_html_body(step, samples, stats, chart_cid, sender=sender, note=note)
    send_email(subject, text_body, html_body, config, images=images)
