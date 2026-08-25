"""
theme.py — Premium dark-mode design system for MyoAdapt UI.

Inspired by Weights & Biases, Linear, and Vercel dashboards:
- Navy-tinted dark canvas (#0B1120) — warmer than pure black
- Blue/cyan/teal/violet pillar colors — one per architectural pillar
- Space Grotesk headings, Inter body, JetBrains Mono numerics
- Bento grid layouts, card-based components, hover micro-interactions
- CSS via st.markdown (no JS, no external dependencies)
"""
from __future__ import annotations

# Color tokens — used in Python code AND CSS
COLORS = {
    "bg": "#0B1120",
    "surface": "#131C31",
    "surface2": "#1C2740",
    "border": "#243049",
    "text": "#F1F5F9",
    "text_muted": "#94A3B8",
    "primary": "#3B82F6",        # Classification pillar
    "accent": "#06B6D4",          # Regression pillar
    "accent2": "#14B8A6",        # Zero-shot pillar
    "violet": "#8B5CF6",         # Domain Adaptation pillar
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#F43F5E",
}

# Pillar colors for the 4-pillar architecture
PILLAR_COLORS = {
    "classification": "#3B82F6",
    "regression": "#06B6D4",
    "zero_shot": "#14B8A6",
    "domain_adaptation": "#8B5CF6",
}

_FONT_LINK = '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">'

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg: #0B1120; --surface: #131C31; --surface-2: #1C2740;
    --border: #243049; --text: #F1F5F9; --text-muted: #94A3B8;
    --primary: #3B82F6; --primary-hover: #60A5FA;
    --accent: #06B6D4; --accent-2: #14B8A6;
    --success: #10B981; --warning: #F59E0B; --danger: #F43F5E; --violet: #8B5CF6;
    --radius: 16px; --radius-sm: 10px;
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text); }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }
code, pre, .stCode { font-family: 'JetBrains Mono', monospace !important; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

/* HERO */
.ma-hero {
    position: relative; border-radius: var(--radius); overflow: hidden;
    padding: 48px 40px; margin-bottom: 28px;
    background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 40%, #0C4A6E 100%);
    border: 1px solid var(--border);
}
.ma-hero::before {
    content: ""; position: absolute; inset: 0;
    background:
        radial-gradient(circle at 15% 20%, rgba(59,130,246,0.25), transparent 40%),
        radial-gradient(circle at 85% 80%, rgba(6,182,212,0.22), transparent 45%),
        radial-gradient(circle at 60% 40%, rgba(139,92,246,0.15), transparent 50%);
    pointer-events: none;
}
.ma-hero .eyebrow {
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    background: rgba(6,182,212,0.15); color: var(--accent);
    font-size: 12px; font-weight: 600; margin-bottom: 16px; position: relative;
    border: 1px solid rgba(6,182,212,0.3);
}
.ma-hero-waveform {
    position: absolute; left: 0; right: 0; bottom: 0; width: 100%; height: 65%;
    -webkit-mask-image: linear-gradient(to bottom, transparent, rgba(0,0,0,.9) 55%, rgba(0,0,0,.55));
    mask-image: linear-gradient(to bottom, transparent, rgba(0,0,0,.9) 55%, rgba(0,0,0,.55));
    pointer-events: none;
}
.ma-hero h1 { font-size: 2.5rem; font-weight: 700; line-height: 1.1; margin: 0 0 12px; position: relative; }
.ma-hero p { font-size: 1.1rem; color: var(--text-muted); max-width: 620px; margin: 0; position: relative; }

/* KPI CARDS */
.ma-kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; margin-bottom: 28px; }
.ma-kpi-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 20px; position: relative; overflow: hidden;
    transition: transform .2s ease, border-color .2s ease;
}
.ma-kpi-card:hover { transform: translateY(-2px); border-color: var(--primary); }
.ma-kpi-card .label { font-size: 11px; font-weight: 500; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
.ma-kpi-card .value { font-family: 'JetBrains Mono', monospace; font-size: 1.7rem; font-weight: 700; line-height: 1; }
.ma-kpi-card .delta { font-size: 12px; margin-top: 6px; color: var(--text-muted); }
.ma-kpi-card .accent-bar { position: absolute; top: 0; left: 0; right: 0; height: 3px; }

/* SECTION HEADER */
.ma-section { margin-top: 1.5rem; margin-bottom: 0.8rem; padding-bottom: 0.4rem; border-bottom: 1px solid var(--border); display: flex; align-items: baseline; gap: 0.5rem; }
.ma-section-num { display: inline-block; background: var(--primary); color: #fff; font-size: 12px; font-weight: 700; padding: 3px 10px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; }
.ma-section-title { font-size: 1.3rem; font-weight: 700; color: var(--text); }

/* BENTO GRID */
.ma-bento { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px; margin-bottom: 28px; }
.ma-tile {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 20px; transition: transform .2s, border-color .2s;
}
.ma-tile:hover { transform: translateY(-2px); border-color: var(--primary); }
.ma-tile h3 { margin: 0 0 8px; font-size: 1.05rem; }
.ma-tile p { margin: 0; color: var(--text-muted); font-size: 0.88rem; line-height: 1.5; }
.ma-tile .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; margin-top: 8px; }
.ma-badge-blue { background: rgba(59,130,246,.12); color: var(--primary); }
.ma-badge-cyan { background: rgba(6,182,212,.12); color: var(--accent); }
.ma-badge-teal { background: rgba(20,184,166,.12); color: var(--accent-2); }
.ma-badge-violet { background: rgba(139,92,246,.12); color: var(--violet); }
.ma-badge-amber { background: rgba(245,158,11,.12); color: var(--warning); }

/* 4-PILLAR ARCHITECTURE */
.ma-arch { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 28px; }
.ma-pillar { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; position: relative; overflow: hidden; transition: transform .2s; }
.ma-pillar:hover { transform: translateY(-3px); }
.ma-pillar::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 4px; }
.ma-pillar.p1::before { background: var(--primary); }
.ma-pillar.p2::before { background: var(--accent); }
.ma-pillar.p3::before { background: var(--accent-2); }
.ma-pillar.p4::before { background: var(--violet); }
.ma-pillar .num { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text-muted); margin-bottom: 6px; }
.ma-pillar h3 { margin: 0 0 6px; font-size: 1.05rem; }
.ma-pillar p { margin: 0; font-size: 0.85rem; color: var(--text-muted); line-height: 1.4; }

/* PAPER CHAIN */
.ma-paper-chain { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 28px; }
.ma-paper-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px 20px; transition: transform .2s, border-color .2s; }
.ma-paper-card:hover { transform: translateY(-2px); border-color: var(--accent); }
.ma-paper-card .meta { font-size: 11px; color: var(--text-muted); margin-bottom: 6px; font-family: 'JetBrains Mono', monospace; }
.ma-paper-card h4 { margin: 0 0 6px; font-size: 0.95rem; line-height: 1.3; }
.ma-paper-card .contrib { font-size: 13px; color: var(--text-muted); line-height: 1.4; }
.ma-paper-card .module { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--accent); margin-top: 6px; }
.ma-paper-card .status { display: inline-block; font-size: 10px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; padding: 2px 8px; border-radius: 999px; margin-left: 8px; vertical-align: middle; }
.ma-status-done { background: rgba(16,185,129,.15); color: var(--success); }
.ma-status-progress { background: rgba(245,158,11,.15); color: var(--warning); }
.ma-status-planned { background: rgba(148,163,184,.15); color: var(--text-muted); }

/* CODE BLOCKS */
.stCode > div, pre { border-radius: 12px !important; border: 1px solid var(--border) !important; background: #0F172A !important; font-size: 13px !important; }

/* FOOTER */
.ma-footer { margin-top: 40px; padding: 20px 0; border-top: 1px solid var(--border); text-align: center; color: var(--text-muted); font-size: 13px; }
.ma-footer a { color: var(--accent); text-decoration: none; }

/* SIDEBAR */
section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }

@media (max-width: 900px) {
    .ma-kpi-grid, .ma-bento, .ma-arch { grid-template-columns: repeat(2, 1fr); }
    .ma-paper-chain { grid-template-columns: 1fr; }
    .ma-hero h1 { font-size: 1.8rem; }
}
</style>
"""


def apply_theme(st_module) -> None:
    """Inject the premium dark-mode design system."""
    # Inject fonts via components.html (handles <link> properly)
    try:
        st_module.components.html(_FONT_LINK, height=0)
    except Exception:
        pass
    st_module.markdown(_CSS, unsafe_allow_html=True)


def sidebar_brand(st_module, version: str = "2.0.0") -> None:
    """Render brand at top of sidebar."""
    st_module.markdown(
        f"""
        <div style="padding: 0.5rem 0;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 700; color: var(--text); letter-spacing: -0.02em;">
                MyoAdapt
            </div>
            <div style="font-size: 0.75rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; margin-top: 2px;">
                v{version} · sEMG platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _emg_trace_path(seed: int, width: int = 1200, height: int = 70, n_points: int = 240,
                     burst_prob: float = 0.12, decay: float = 0.90) -> str:
    """Generate an SVG polyline path resembling a real sEMG channel: a
    noisy baseline with sudden amplitude bursts that decay exponentially
    (roughly how a muscle contraction reads on one electrode), not a
    clean sine wave. Deterministic per seed so the page doesn't jitter
    on every rerun.
    """
    import random
    rng = random.Random(seed)
    y_mid = height / 2
    envelope = 0.08
    points = []
    for i in range(n_points):
        if rng.random() < burst_prob:
            envelope = min(1.0, envelope + rng.uniform(0.4, 0.9))
        envelope *= decay + rng.uniform(-0.02, 0.02)
        envelope = max(0.05, envelope)
        sample = rng.uniform(-1, 1) * envelope
        x = (i / (n_points - 1)) * width
        y = y_mid + sample * (height / 2 - 4)
        points.append(f"{x:.1f},{y:.1f}")
    return "M" + " L".join(points)


def waveform_motif(n_channels: int = 4) -> str:
    """Return an SVG of several stacked, semi-transparent sEMG-style
    channel traces — the hero's signature element, grounded in what the
    platform actually processes rather than an arbitrary gradient."""
    colors = [COLORS["primary"], COLORS["accent"], COLORS["accent2"], COLORS["violet"]]
    height_per = 70
    total_h = height_per * n_channels
    lines = []
    for ch in range(n_channels):
        path = _emg_trace_path(seed=1000 + ch, height=height_per)
        color = colors[ch % len(colors)]
        lines.append(
            f'<g transform="translate(0,{ch * height_per})">'
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.4" '
            f'opacity="0.35" stroke-linejoin="round" stroke-linecap="round"/></g>'
        )
    return (
        f'<svg class="ma-hero-waveform" viewBox="0 0 1200 {total_h}" '
        f'preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">'
        + "".join(lines) + "</svg>"
    )


def hero(eyebrow: str = "", title: str = "", subtitle: str = "") -> str:
    """Return hero HTML."""
    return f"""
    <div class="ma-hero">
        {waveform_motif()}
        <span class="eyebrow">{eyebrow}</span>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """


def section_header(num: str, title: str) -> str:
    """Return section header HTML."""
    return f"""
    <div class="ma-section">
        <span class="ma-section-num">{num}</span>
        <span class="ma-section-title">{title}</span>
    </div>
    """


def info_card(title: str, body: str, badge: str = "", badge_kind: str = "cyan") -> str:
    """Return info card HTML (bento tile)."""
    badge_html = f'<span class="badge ma-badge-{badge_kind}">{badge}</span>' if badge else ""
    return f"""
    <div class="ma-tile">
        <h3>{title}</h3>
        <p>{body}</p>
        {badge_html}
    </div>
    """


def badge(text: str, kind: str = "cyan") -> str:
    """Return badge HTML."""
    return f'<span class="badge ma-badge-{kind}">{text}</span>'


def footer() -> str:
    """Return footer HTML."""
    return """
    <div class="ma-footer">
        <strong>MyoAdapt</strong> · Open-source sEMG pattern recognition · Apache 2.0<br/>
        <a href="https://github.com/Qussai-BME/MyoAdapt" target="_blank">GitHub</a> ·
        <a href="https://orcid.org/0009-0000-7667-1992" target="_blank">ORCID</a> ·
        CPU-native · Edge-deployable · Audit-ready
    </div>
    """


def spacer(sm: bool = True) -> str:
    """Return spacer div."""
    return '<div style="height:0.8rem;"></div>' if sm else '<div style="height:1.5rem;"></div>'


# Backward compat alias
PALETTE = COLORS
