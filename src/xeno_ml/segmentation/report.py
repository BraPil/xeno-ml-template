from pathlib import Path
from datetime import datetime
import base64, io, sys

import numpy as np
from PIL import Image, ImageOps
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parents[3]          # project root
TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(ROOT / "templates"), autoescape=True
)

try:
    import weasyprint
    _PDF_OK = True
except Exception:                                   # missing GTK / Pango
    weasyprint = None
    _PDF_OK = False

# ───────────────────────── helpers ─────────────────────────
def _overlay(img: np.ndarray, mask: np.ndarray) -> Image.Image:
    base = Image.fromarray(img).convert("RGB") if img.ndim == 2 else Image.fromarray(img)
    red = Image.new("RGB", base.size, (255, 0, 0))
    alpha = Image.fromarray((mask > 0).astype("uint8") * 120).resize(base.size)
    base.paste(red, mask=alpha)
    return ImageOps.fit(base, (120, 120))

# ───────────────────────── main API ────────────────────────
def build_report(images, masks, out_path: Path) -> Path:
    """Always writes HTML; writes PDF too if WeasyPrint can do it.
    Returns Path to the HTML (so caller can log it)."""
    rows = []
    for idx, (img, mask) in enumerate(zip(images, masks), start=1):
        thumb = _overlay(img, mask)
        buf = io.BytesIO(); thumb.save(buf, format="PNG")
        rows.append(dict(
            name=f"img_{idx}",
            area=int((mask > 0).sum()),
            thumb="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        ))

    html = TEMPLATE_ENV.get_template("report.html").render(
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_images=len(images),
        masks_found=sum(r["area"] > 0 for r in rows),
        rows=rows,
    )

    html_path = out_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")

    if _PDF_OK:
        try:
            weasyprint.HTML(string=html).write_pdf(out_path)
        except Exception as exc:
            print("⚠️  PDF generation failed:", exc, file=sys.stderr)

    return html_path   # always exists
