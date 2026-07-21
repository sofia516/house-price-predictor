"""
Build a self-contained static site (index.html) from the project artifacts,
then deploy it with the `website_deploy` tool.

Reads:
  artifacts/leaderboard.csv
  artifacts/feature_importance.csv
  artifacts/pred_vs_actual.png
  eda_overview.png

Writes:
  site/index.html   (self-contained, base64-embedded images)
"""
from __future__ import annotations

import base64
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
SITE = ROOT / "site"
SITE.mkdir(exist_ok=True)


def b64_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def bar_chart_svg(rows: list[tuple[str, float]], max_bars: int = 12) -> str:
    """Return an inline SVG horizontal bar chart from a list of (label, value) pairs."""
    rows = rows[:max_bars]
    if not rows:
        return "<svg></svg>"
    max_v = max(v for _, v in rows) or 1.0
    bar_h, gap, pad_l, pad_r, pad_t = 26, 8, 220, 60, 20
    w = 720
    h = pad_t + len(rows) * (bar_h + gap) + 10
    parts = [
        f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
        f'style="font-family: -apple-system, system-ui, sans-serif; font-size: 12px;">'
    ]
    for i, (label, v) in enumerate(rows):
        y = pad_t + i * (bar_h + gap)
        bar_w = (w - pad_l - pad_r) * (v / max_v)
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + bar_h/2 + 4}" text-anchor="end" '
            f'fill="#1f2937">{label}</text>'
        )
        parts.append(
            f'<rect x="{pad_l}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" '
            f'fill="url(#g)" rx="4"/>'
        )
        parts.append(
            f'<text x="{pad_l + bar_w + 6:.1f}" y="{y + bar_h/2 + 4}" '
            f'fill="#374151">${v:,.0f}</text>'
        )
    parts.append(
        '<defs><linearGradient id="g" x1="0" x2="1">'
        '<stop offset="0%" stop-color="#1f4e79"/>'
        '<stop offset="100%" stop-color="#2e8b57"/>'
        '</linearGradient></defs>'
    )
    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    leaderboard = pd.read_csv(ROOT / "artifacts" / "leaderboard.csv")
    importance  = pd.read_csv(ROOT / "artifacts" / "feature_importance.csv")

    eda_b64 = b64_png(ROOT / "eda_overview.png")
    pva_b64 = b64_png(ROOT / "artifacts" / "pred_vs_actual.png")

    # leaderboard rows
    lb_rows = ""
    for _, r in leaderboard.iterrows():
        medal = {0: "🥇", 1: "🥈", 2: "🥉"}.get(_, "")
        lb_rows += (
            f"<tr>"
            f"<td>{medal} {r['Model']}</td>"
            f"<td>${r['Test RMSE']:,.0f}</td>"
            f"<td>${r['Test MAE']:,.0f}</td>"
            f"<td>{r['Test R2']:.3f}</td>"
            f"</tr>"
        )

    rows = list(zip(
        importance["feature"].tolist(),
        importance["importance"].astype(float).tolist(),
    ))
    chart_svg = bar_chart_svg(rows, max_bars=12)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>House Price Predictor — California Housing</title>
<meta name="description" content="End-to-end regression project predicting California median house values with XGBoost. Test RMSE $42,091, R² 0.868.">
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #111827;
    background: #f8fafc;
    line-height: 1.6;
  }}
  a {{ color: #1f4e79; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
  .hero {{
    background: linear-gradient(135deg, #1f4e79 0%, #2e8b57 100%);
    color: white;
    padding: 80px 24px 100px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.1);
  }}
  .hero h1 {{ font-size: 48px; margin: 0 0 16px; font-weight: 800; letter-spacing: -1px; }}
  .hero p {{ font-size: 18px; opacity: 0.92; max-width: 700px; margin: 0 auto 32px; }}
  .badges {{ display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; margin-bottom: 32px; }}
  .badge {{
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 13px;
    backdrop-filter: blur(8px);
  }}
  .cta {{
    display: inline-block;
    background: white;
    color: #1f4e79;
    padding: 14px 32px;
    border-radius: 999px;
    font-weight: 700;
    margin: 8px;
    transition: transform 0.15s;
  }}
  .cta:hover {{ transform: translateY(-2px); text-decoration: none; }}
  .cta.secondary {{ background: transparent; color: white; border: 2px solid white; }}
  h2 {{
    font-size: 28px;
    margin: 56px 0 24px;
    color: #1f2937;
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 8px;
  }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  @media (max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} .hero h1 {{ font-size: 32px; }} }}
  .card {{
    background: white;
    padding: 24px;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border: 1px solid #e5e7eb;
  }}
  table {{ width: 100%; border-collapse: collapse; }}
  th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid #f1f5f9; }}
  th {{ background: #f1f5f9; font-weight: 600; color: #475569; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }}
  td {{ font-size: 14px; }}
  img {{ max-width: 100%; border-radius: 8px; display: block; }}
  .stat-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }}
  @media (max-width: 768px) {{ .stat-row {{ grid-template-columns: repeat(2, 1fr); }} }}
  .stat {{
    background: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #e5e7eb;
  }}
  .stat .num {{ font-size: 28px; font-weight: 800; color: #1f4e79; }}
  .stat .lbl {{ font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }}
  footer {{ text-align: center; padding: 40px 24px; color: #6b7280; font-size: 14px; border-top: 1px solid #e5e7eb; margin-top: 80px; background: white; }}
  code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 13px; }}
</style>
</head>
<body>

<section class="hero">
  <h1>🏠 House Price Predictor</h1>
  <p>End-to-end regression on California Housing — feature engineering, model zoo, hyperparameter tuning, and stacking. Tuned XGBoost hits <strong>$42,091 RMSE</strong> on the held-out test set (R² = 0.868).</p>
  <div class="badges">
    <span class="badge">🐍 scikit-learn</span>
    <span class="badge">🚀 XGBoost</span>
    <span class="badge">📊 Regression</span>
    <span class="badge">🧪 Cross-validated</span>
    <span class="badge">⚡ 20,640 rows</span>
  </div>
  <a href="https://github.com" class="cta">⭐ View on GitHub</a>
  <a href="#results" class="cta secondary">See the results ↓</a>
</section>

<div class="container">

  <div class="stat-row">
    <div class="stat"><div class="num">$42k</div><div class="lbl">Test RMSE</div></div>
    <div class="stat"><div class="num">$27k</div><div class="lbl">Test MAE</div></div>
    <div class="stat"><div class="num">0.87</div><div class="lbl">R² Score</div></div>
    <div class="stat"><div class="num">6</div><div class="lbl">Models Compared</div></div>
  </div>

  <h2 id="results">🏆 Model leaderboard</h2>
  <div class="card">
    <table>
      <thead><tr><th>Model</th><th>Test RMSE</th><th>Test MAE</th><th>R²</th></tr></thead>
      <tbody>
        {lb_rows}
      </tbody>
    </table>
    <p style="color: #6b7280; font-size: 13px; margin-top: 16px;">
      Tuned XGBoost beat the stacked ensemble. Trees crush linear here because the
      relationship is full of interactions and non-linearities, and linear models
      can't represent the well-known $500k price cap in the dataset.
    </p>
  </div>

  <h2>🔍 Feature importance</h2>
  <div class="card">
    {chart_svg}
    <p style="color: #6b7280; font-size: 13px; margin-top: 16px;">
      Permutation importance on the test set. <strong>Location dominates</strong> —
      latitude, longitude, and distance to SF explain more variance than any
      other single feature. Income is the only non-geo feature in the top 3.
    </p>
  </div>

  <h2>📈 Diagnostics</h2>
  <div class="grid">
    <div class="card">
      <h3 style="margin-top:0">Predicted vs actual (test set)</h3>
      <img src="data:image/png;base64,{pva_b64}" alt="Predicted vs actual scatter">
    </div>
    <div class="card">
      <h3 style="margin-top:0">EDA overview</h3>
      <img src="data:image/png;base64,{eda_b64}" alt="EDA plots">
    </div>
  </div>

  <h2>🛠️ Stack & workflow</h2>
  <div class="card">
    <ol>
      <li><strong>EDA</strong> — distributions, correlations, geographic scatter, price-by-proximity</li>
      <li><strong>Cleaning</strong> — median imputation of 207 missing <code>total_bedrooms</code>, duplicates, sanity checks</li>
      <li><strong>Feature engineering</strong> — 13 new features: per-household ratios, log-transforms, <code>AgeSquared</code>, distance to LA / SF / coast, KMeans cluster on lat/lon, ordinal coast</li>
      <li><strong>5-fold CV</strong> across 6 model families: Linear, Ridge, Lasso, RF, GBR, XGBoost</li>
      <li><strong>Randomized search</strong> on XGBoost (15 iterations)</li>
      <li><strong>Stacking</strong> — Ridge + RF + GBR + XGB → Ridge meta-learner</li>
      <li><strong>Test-set evaluation</strong> on a held-out 20%</li>
    </ol>
  </div>

  <h2>🚀 Run it yourself</h2>
  <div class="card">
<pre style="background: #0f172a; color: #e2e8f0; padding: 20px; border-radius: 8px; overflow-x: auto; font-size: 13px;"><code>git clone https://github.com/&lt;you&gt;/house-price-predictor.git
cd house-price-predictor
pip install -r requirements.txt
python house_price_predictor.py        # full pipeline (~12 min)
streamlit run app.py                   # interactive UI</code></pre>
  </div>

</div>

<footer>
  <p>MIT License · Built as a learning project on regression, feature engineering, and model evaluation</p>
</footer>

</body>
</html>"""

    (SITE / "index.html").write_text(html, encoding="utf-8")
    print(f"Wrote {SITE / 'index.html'}  ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
