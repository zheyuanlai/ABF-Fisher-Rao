#!/usr/bin/env python3
"""Build the campaign report artifact (self-contained HTML) from stored results."""
from __future__ import annotations
import base64, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
from rcwfr.campaign import paired_bootstrap, rel_change


def jl(p):
    with open(p) as f:
        return json.load(f)


def img(name, alt):
    p = HERE / f"{name}.png"
    if not p.exists():
        return ""
    b = base64.b64encode(p.read_bytes()).decode()
    return f'<img alt="{alt}" src="data:image/png;base64,{b}">'


def load_confirm(sysname):
    for t in ("_cal", ""):
        p = RES / "confirm" / f"{sysname}{t}.json"
        if p.exists():
            return jl(p)
    return None


PRETTY = {
    "wfr_oracle": ("RC-WFR, oracle lift", "oracle"),
    "wfr_scaled": ("RC-WFR, model lift", "wfr"),
    "wfr_flow": ("RC-WFR, flow + FR", "wfr"),
    "wfr_flow_cnt": ("RC-WFR, flow + count", "wfr"),
    "wfr_flow_w": ("RC-WFR, flow, no FR", "wfr"),
    "wfr_gmm": ("RC-WFR, GMM score", "wfr"),
    "wfr": ("RC-WFR, SDE + FR", "wfr"),
    "wfr_anneal": ("RC-WFR, annealed", "wfr"),
    "w_only": ("Wasserstein only", "wfr"),
    "fr_only": ("Fisher-Rao only", "wfr"),
    "w_count": ("W + count balancing", "wfr"),
    "w_sham": ("W + sham churn", "ctrl"),
    "ti_cold": ("stratified TI, cold", "clas"),
    "ti_warm": ("stratified TI, warm", "oracle"),
    "reti_cold": ("RE-TI, cold", "clas"),
    "reti_warm": ("RE-TI, warm", "oracle"),
    "abf": ("ABF", "adapt"),
    "shus": ("SHUS / ABP", "adapt"),
    "unbiased": ("unbiased MD", "ctrl"),
}


def arm_table(d, baseline="ti_cold"):
    fl, arms = d["floor"], d["arms"]
    rows = []
    for k, v in arms.items():
        IF = np.asarray(v["I_F"], float)
        m, lo, hi = paired_bootstrap(IF)
        eF = float(np.median(np.asarray(v["e_F_final"], float)))
        dm, dlo, dhi = paired_bootstrap(rel_change(v["I_F"], arms[baseline]["I_F"])) \
            if baseline in arms else (np.nan,) * 3
        rows.append((k, m, lo, hi, eF, eF / fl,
                     float(np.median(np.asarray(v["chan"], float))), dm, dlo, dhi))
    rows.sort(key=lambda r: r[1])
    out = ['<div class="tw"><table><thead><tr>'
           '<th>arm</th><th class="n">I<sub>F</sub></th>'
           '<th class="n">95% CI</th><th class="n">e<sub>F</sub></th>'
           '<th class="n">&times; floor</th><th class="n">channel err</th>'
           f'<th class="n">vs {PRETTY.get(baseline, (baseline,))[0]}</th>'
           '</tr></thead><tbody>']
    for k, m, lo, hi, eF, rat, ch, dm, dlo, dhi in rows:
        name, kind = PRETTY.get(k, (k, ""))
        star = " <sup>*</sup>" if kind == "oracle" else ""
        sig = "pos" if (dhi < 0) else ("neg" if dlo > 0 else "tie")
        delta = ("&mdash;" if k == baseline else
                 f'<span class="d {sig}">{100*dm:+.0f}%</span>')
        out.append(f'<tr class="k-{kind}"><td>{name}{star}</td>'
                   f'<td class="n">{m:.5f}</td>'
                   f'<td class="n sm">[{lo:.5f}, {hi:.5f}]</td>'
                   f'<td class="n">{eF:.5f}</td><td class="n">{rat:.1f}</td>'
                   f'<td class="n">{ch:.3f}</td><td class="n">{delta}</td></tr>')
    out.append("</tbody></table></div>")
    return "\n".join(out), fl


def torsion_rows():
    ds = [jl(f) for f in sorted((RES / "torsion").glob("torsion_scaling*.json"))]
    if not ds:
        return "", []
    Ls = sorted({float(k) for d in ds for k in d})
    rows = []
    for L in Ls:
        best = {}
        for pre in ("wfr", "abf", "ti_cold", "reti_cold"):
            vals, keyd = [], None
            for d in ds:
                if str(L) not in d:
                    continue
                for k, v in d[str(L)]["arms"].items():
                    if k.startswith(pre):
                        vals.append((float(np.median(v["I_F"])), v["I_F"]))
            if vals:
                best[pre] = min(vals, key=lambda t: t[0])
        if "wfr" in best and "abf" in best:
            dm, dlo, dhi = paired_bootstrap(rel_change(best["wfr"][1], best["abf"][1]))
        else:
            dm = dlo = dhi = np.nan
        rows.append((L, {k: v[0] for k, v in best.items()}, (dm, dlo, dhi)))
    html = ['<div class="tw"><table><thead><tr><th class="n">CV length L</th>'
            '<th class="n">RC-WFR</th><th class="n">ABF</th>'
            '<th class="n">stratified TI</th><th class="n">RE-TI</th>'
            '<th class="n">RC-WFR vs ABF</th></tr></thead><tbody>']
    for L, b, (dm, dlo, dhi) in rows:
        sig = "pos" if dhi < 0 else ("neg" if dlo > 0 else "tie")
        html.append(
            f'<tr><td class="n">{L:g}</td>'
            + "".join(f'<td class="n">{b.get(k, float("nan")):.5f}</td>'
                      for k in ("wfr", "abf", "ti_cold", "reti_cold"))
            + f'<td class="n"><span class="d {sig}">{100*dm:+.0f}%</span>'
              f'<span class="sm"> [{100*dlo:+.0f}, {100*dhi:+.0f}]</span></td></tr>')
    html.append("</tbody></table></div>")
    return "\n".join(html), rows


def mspec_table():
    p = RES / "mspec" / "CHANNEL_mspec.json"
    if not p.exists():
        return ""
    d = jl(p)
    html = ['<div class="tw"><table><thead><tr><th class="n">spectator dofs m</th>'
            '<th class="n">RE acceptance</th><th class="n">RC-WFR</th>'
            '<th class="n">stratified TI</th><th class="n">RE-TI</th>'
            '<th class="n">RC-WFR vs RE-TI</th></tr></thead><tbody>']
    for m in sorted(d, key=int):
        r = d[m]; a = r["arms"]
        g = lambda k: float(np.median(np.asarray(a[k]["I_F_rel"], float)))
        dm, dlo, dhi = r["cmp"]["wfr_vs_reti_cold_M256"]
        sig = "pos" if dhi < 0 else ("neg" if dlo > 0 else "tie")
        html.append(f'<tr><td class="n">{int(m)}</td>'
                    f'<td class="n">{a["reti_cold_M256"]["ex_accept"]:.3f}</td>'
                    f'<td class="n">{g("wfr"):.4f}</td>'
                    f'<td class="n">{g("ti_cold"):.4f}</td>'
                    f'<td class="n">{g("reti_cold_M256"):.4f}</td>'
                    f'<td class="n"><span class="d {sig}">{100*dm:+.0f}%</span></td></tr>')
    html.append("</tbody></table></div>")
    return "\n".join(html)


if __name__ == "__main__":
    eb, ch = load_confirm("EB"), load_confirm("CHANNEL")
    print("EB arms:", len(eb["arms"]) if eb else 0,
          "| CHANNEL arms:", len(ch["arms"]) if ch else 0)
    t, rows = torsion_rows()
    print("torsion L values:", [r[0] for r in rows])


CSS = """
:root{
  --bg:#F4F6F6; --surface:#FFFFFF; --surface-2:#EDF1F1;
  --ink:#101718; --ink-2:#39494C; --muted:#5F7074; --rule:#D8E1E1;
  --accent:#C1440E;          /* RC-WFR, the series colour used in every figure */
  --clas:#0B6E8C;            /* classical stratification */
  --adapt:#1D2426;           /* adaptive biasing */
  --oracle:#849699;
  --pos:#1C6B4B; --neg:#9A3412; --tie:#6B7A7D;
  --shadow:0 1px 2px rgba(16,23,24,.05), 0 8px 24px -16px rgba(16,23,24,.28);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#0C1113; --surface:#131A1C; --surface-2:#182124;
    --ink:#E7EDED; --ink-2:#BCC9CB; --muted:#8DA0A3; --rule:#233032;
    --accent:#F0803C; --clas:#4FB6D4; --adapt:#D6DEDF; --oracle:#7B8B8E;
    --pos:#5FCF9E; --neg:#F0906A; --tie:#8DA0A3;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 30px -18px rgba(0,0,0,.8);
  }
}
:root[data-theme="dark"]{
  --bg:#0C1113; --surface:#131A1C; --surface-2:#182124;
  --ink:#E7EDED; --ink-2:#BCC9CB; --muted:#8DA0A3; --rule:#233032;
  --accent:#F0803C; --clas:#4FB6D4; --adapt:#D6DEDF; --oracle:#7B8B8E;
  --pos:#5FCF9E; --neg:#F0906A; --tie:#8DA0A3;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 30px -18px rgba(0,0,0,.8);
}

*{box-sizing:border-box}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:"Source Sans 3","Source Sans Pro",-apple-system,Segoe UI,sans-serif;
  font-size:16.5px; line-height:1.62; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1180px; margin:0 auto; padding:0 clamp(18px,4vw,48px) 96px}
.col{max-width:70ch}
h1,h2,h3{font-family:Spectral,Georgia,"Times New Roman",serif; font-weight:600;
  text-wrap:balance; line-height:1.18; margin:0}
h1{font-size:clamp(2.1rem,5vw,3.05rem); letter-spacing:-.015em}
h2{font-size:clamp(1.35rem,2.6vw,1.72rem); margin:0 0 .1em}
h3{font-size:1.06rem; font-weight:600; margin:2.1em 0 .35em}
p{margin:0 0 1.05em}
a{color:var(--accent)}
strong{font-weight:600}
code,.n,.mono{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
code{font-size:.9em; background:var(--surface-2); padding:.1em .34em; border-radius:3px}

/* ---- masthead ---- */
header{padding:clamp(52px,9vw,104px) 0 0}
.eyebrow{font-family:"IBM Plex Mono",monospace; font-size:.72rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted); margin:0 0 1.1em}
.lede{font-size:1.16rem; color:var(--ink-2); max-width:62ch; margin-top:1.1em}
.meta{display:flex; flex-wrap:wrap; gap:.5em 1.6em; margin-top:2.2em;
  font-family:"IBM Plex Mono",monospace; font-size:.74rem; color:var(--muted)}
.meta b{color:var(--ink-2); font-weight:500}

/* ---- the estimator-floor rule: the page's structural motif ---- */
.floor{border:0; border-top:1.5px dotted var(--rule); margin:clamp(48px,7vw,84px) 0 0}
section{padding-top:clamp(30px,4vw,44px)}
.sec-no{font-family:"IBM Plex Mono",monospace; font-size:.72rem; letter-spacing:.12em;
  color:var(--accent); display:block; margin-bottom:.7em}

/* ---- verdict cards ---- */
.cards{display:grid; gap:14px; margin:2.4em 0 0;
  grid-template-columns:repeat(auto-fit,minmax(238px,1fr))}
.card{background:var(--surface); border:1px solid var(--rule); border-radius:2px;
  padding:20px 20px 18px; box-shadow:var(--shadow); position:relative;
  border-top:2px solid var(--accent)}
.card.c-clas{border-top-color:var(--clas)}
.card.c-mix{border-top-color:var(--oracle)}
.card h3{margin:0 0 .5em; font-size:.79rem; font-family:"IBM Plex Mono",monospace;
  letter-spacing:.09em; text-transform:uppercase; color:var(--muted); font-weight:500}
.big{font-family:Spectral,Georgia,serif; font-size:2.15rem; line-height:1;
  letter-spacing:-.02em; display:block; margin-bottom:.28em}
.card.win .big{color:var(--accent)}
.card.lose .big{color:var(--clas)}
.card p{font-size:.9rem; color:var(--ink-2); margin:0}

/* ---- tables ---- */
.tw{overflow-x:auto; margin:1.6em 0 .5em; border:1px solid var(--rule);
  border-radius:2px; background:var(--surface)}
table{border-collapse:collapse; width:100%; font-size:.85rem}
th,td{padding:7px 13px; text-align:left; border-bottom:1px solid var(--rule);
  white-space:nowrap}
thead th{font-family:"IBM Plex Mono",monospace; font-size:.68rem; font-weight:500;
  letter-spacing:.07em; text-transform:uppercase; color:var(--muted);
  background:var(--surface-2); position:sticky; top:0}
tbody tr:last-child td{border-bottom:0}
.n{font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums;
  text-align:right}
.sm{font-size:.76rem; color:var(--muted)}
.d{font-weight:600}
.d.pos{color:var(--pos)} .d.neg{color:var(--neg)} .d.tie{color:var(--tie)}
tr.k-wfr td:first-child{border-left:3px solid var(--accent); font-weight:600}
tr.k-clas td:first-child{border-left:3px solid var(--clas)}
tr.k-adapt td:first-child{border-left:3px solid var(--adapt)}
tr.k-oracle td:first-child{border-left:3px solid var(--oracle); color:var(--muted)}
tr.k-ctrl td:first-child{border-left:3px solid transparent; color:var(--muted)}
.caption{font-size:.83rem; color:var(--muted); max-width:78ch; margin:.6em 0 0}

/* ---- figures ---- */
figure{margin:2.2em 0 1.2em}
/* the figures are rendered on white by matplotlib; present them as a plate with a
   little breathing room so they read as inset artwork rather than a hole in a dark
   page, rather than recolouring scientific output */
figure img{width:100%; height:auto; display:block; border:1px solid var(--rule);
  border-radius:2px; background:#FFFFFF; padding:8px}
figcaption{font-size:.83rem; color:var(--muted); margin-top:.7em; max-width:82ch}
figcaption b{color:var(--ink-2); font-weight:600}

/* ---- callout ---- */
.claim{border-left:3px solid var(--accent); background:var(--surface);
  padding:18px 22px; margin:1.9em 0; box-shadow:var(--shadow)}
.claim p{margin:0; font-family:Spectral,Georgia,serif; font-size:1.06rem;
  line-height:1.55}
.claim p + p{margin-top:.7em}

/* ---- findings list ---- */
.find{list-style:none; padding:0; margin:1.4em 0 0}
.find li{display:grid; grid-template-columns:auto 1fr; gap:0 16px;
  padding:14px 0; border-top:1px solid var(--rule)}
.find li:first-child{border-top:0}
.tag{font-family:"IBM Plex Mono",monospace; font-size:.7rem; letter-spacing:.06em;
  padding:2px 7px; border-radius:2px; height:fit-content; white-space:nowrap;
  border:1px solid var(--rule); color:var(--muted); background:var(--surface-2)}
.tag.y{color:var(--pos); border-color:var(--pos)}
.tag.n{color:var(--neg); border-color:var(--neg)}
.find b{display:block; margin-bottom:.15em}
.find span{font-size:.93rem; color:var(--ink-2)}

/* ---- regime table ---- */
.regime td:first-child{white-space:normal; max-width:38ch}
footer{margin-top:clamp(56px,8vw,92px); padding-top:26px;
  border-top:1.5px dotted var(--rule); font-size:.83rem; color:var(--muted)}
:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
@media (prefers-reduced-motion:no-preference){
  .card{transition:transform .18s ease}
  .card:hover{transform:translateY(-1px)}
}
"""


def build():
    eb, ch = load_confirm("EB"), load_confirm("CHANNEL")
    eb_tbl, eb_fl = arm_table(eb, "ti_cold")
    ch_tbl, ch_fl = arm_table(ch, "reti_cold")
    tors_tbl, tors_rows = torsion_rows()
    ms_tbl = mspec_table()

    def g(d, k, field="I_F"):
        return float(np.median(np.asarray(d["arms"][k][field], float)))

    def pair(d, a, b):
        m, lo, hi = paired_bootstrap(rel_change(d["arms"][a]["I_F"], d["arms"][b]["I_F"]))
        return 100 * m, 100 * lo, 100 * hi

    eb_vs_ti = pair(eb, "wfr_flow", "ti_cold")
    eb_vs_abf = pair(eb, "wfr_flow", "abf")
    ch_vs_reti = pair(ch, "wfr_flow", "reti_cold")
    ch_vs_ti = pair(ch, "wfr_flow", "ti_cold")
    ch_vs_abf = pair(ch, "wfr_flow", "abf")
    # highlight L = 24: the longest domain at which every arm's own knobs were screened
    L_hi = next((r for r in tors_rows if abs(r[0] - 24.0) < 1e-9), tors_rows[-1])

    head = f"""<title>Does reaction-coordinate WFR beat adaptive biasing?</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@400;600&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>"""

    body = f"""
<div class="wrap">
<header class="col">
  <p class="eyebrow">Campaign record &middot; reaction-coordinate-wfr</p>
  <h1>Does reaction-coordinate WFR beat adaptive biasing?</h1>
  <p class="lede">A bias-free free-energy method that moves replicas through
  reaction-coordinate space by a Wasserstein&ndash;Fisher&ndash;Rao flow, keeps physical
  conditional sampling on the fibers, and reconstructs <span class="mono">F</span> by
  thermodynamic integration. No ABF, no OPES, no learned bias. This is what it can and
  cannot do, at matched force-evaluation cost.</p>
  <div class="meta">
    <span><b>4</b> systems</span><span><b>19</b> method arms</span>
    <span><b>32</b> confirmation seeds</span>
    <span><b>25.6M</b> force evaluations per arm</span>
    <span>estimator floor <b>{eb_fl:.4f}</b> / <b>{ch_fl:.4f}</b></span>
  </div>
</header>

<hr class="floor">
<section>
  <div class="col">
    <span class="sec-no">01 &mdash; Verdict</span>
    <h2>Yes against adaptive biasing, in a regime you can identify in advance</h2>
    <p>Two things had to be got right before the method could be judged at all, and
    both were found by the campaign rather than assumed. First, the <em>stochastic</em>
    Wasserstein step is the wrong one: replacing it with the deterministic probability
    flow <code>Z &larr; Z &minus; &kappa;&Delta;&tau;&nabla;log p&#770;(Z)</code> changes
    the error by up to an order of magnitude, because the flow velocity vanishes as
    <code>p &rarr; u</code>, so its hysteresis self-annihilates. Second, deterministic
    transport and Fisher&ndash;Rao resampling are incompatible without a small
    resample&ndash;move jitter &mdash; clones follow identical trajectories and the
    ensemble collapses.</p>
  </div>
  <div class="cards">
    <div class="card win"><h3>vs replica-exchange TI &mdash; hidden channel</h3>
      <span class="big">{ch_vs_reti[0]:+.0f}%</span>
      <p>against a cold-start Hamiltonian-exchange baseline screened over 12
      configurations, on a fiber whose slow mode has a localized gateway.
      95% CI [{ch_vs_reti[1]:+.0f}, {ch_vs_reti[2]:+.0f}]. Same arm vs stratified TI:
      {ch_vs_ti[0]:+.0f}%; vs ABF: {ch_vs_abf[0]:+.0f}%.</p></div>
    <div class="card win"><h3>vs ABF &mdash; long CV domain</h3>
      <span class="big">{100*L_hi[2][0]:+.0f}%</span>
      <p>at <span class="mono">L&nbsp;=&nbsp;{L_hi[0]:g}</span>, and the margin grows
      monotonically with domain length &mdash; at <span class="mono">L&nbsp;=&nbsp;3</span>
      RC-WFR is 3&times; <em>worse</em>. Exactly the crossover the
      <span class="mono">O(L)</span> vs <span class="mono">O(L&sup2;)</span> argument
      predicts.</p></div>
    <div class="card lose c-clas"><h3>vs stratified TI &mdash; easy fiber</h3>
      <span class="big">{eb_vs_ti[0]:+.0f}%</span>
      <p>plain cold-start fixed-window TI still wins where the fiber is unimodal and
      fast: it has no transport, so it has no lift bias. The same arm still beats ABF
      there by {abs(eb_vs_abf[0]):.0f}%.</p></div>
  </div>
  <div class="col">
    <div class="claim">
      <p>Any move that changes <span class="mono">&xi;(q)</span> without knowing
      <span class="mono">F</span> cannot be Metropolis-corrected: the acceptance ratio
      for the target <span class="mono">u(z)&nu;<sup>&xi;</sup>(dq|z)</span> contains
      <span class="mono">exp(+&beta;F(&xi;(q)))</span>.</p>
      <p>Replica exchange escapes this by swapping between two <em>occupied</em>
      windows, where the unknown weights cancel identically. RC-WFR instead moves
      unconditionally and does not correct &mdash; buying CV transport at the price of a
      hysteresis bias set by <span class="mono">&kappa;&tau;<sub>fiber</sub></span>
      summed over every fiber mode it drags, including the slowest one, which is
      precisely the mode that made physical CV transport slow to begin with.</p>
    </div>
    <p>So RC-WFR converts a <em>convergence</em> problem into a <em>bias</em> problem.
    More compute fixes the first and not the second. The deterministic flow mitigates
    this but does not repeal it. The one component that is entirely free of it is
    Fisher&ndash;Rao: selection copies a walker together with its fiber configuration and
    drags nothing. That is why the best configurations use a <em>small</em>
    <span class="mono">&kappa;</span> and a <em>large</em> <span class="mono">&theta;</span>
    &mdash; minimum dragging, maximum birth&ndash;death &mdash; and why removing
    Fisher&ndash;Rao costs a factor 2.4&ndash;2.5.</p>
  </div>
</section>

<hr class="floor">
<section>
  <div class="col">
    <span class="sec-no">02 &mdash; Mechanism</span>
    <h2>W discovers, FR establishes &mdash; and the marginal claim is exactly right</h2>
    <p>Before any free energy was computed, the particle operators were checked against
    the Wasserstein&ndash;Fisher&ndash;Rao PDE they are supposed to realize. The
    Wasserstein arm tracks it to 0.28% median relative deviation and the combined arm to
    3.8%. Fisher&ndash;Rao alone does not track it at all &mdash; and that mismatch is
    the point, not a bug: the Eulerian density is positive everywhere, while particles
    cannot move mass to where there are none. Its support width was unchanged to four
    decimals over 500 events at two bandwidths.</p>
  </div>
  <figure>{img("fig1_mechanism", "KL to uniform versus time for W, FR and W+FR against the PDE; and time to uniformity versus domain size on log axes")}
  <figcaption><b>The complementarity is quantitative.</b> Time to reach
  <span class="mono">KL &lt; 0.05</span> scales as <span class="mono">L&sup2;</span> for
  Wasserstein alone (ratios 4.4, 4.05, 4.00) and as <span class="mono">L</span> for
  W+FR (2.08, 1.89, 2.02): a reaction&ndash;diffusion front of speed
  <span class="mono">~2&radic;(&kappa;&lambda;)</span> replacing diffusive relaxation.
  Fisher&ndash;Rao alone never converges. This is what predicts an advantage over ABF
  that <em>grows</em> with CV domain length, since ABF's CV equilibration is also
  diffusive.</figcaption></figure>
</section>

<hr class="floor">
<section>
  <div class="col">
    <span class="sec-no">03 &mdash; The lift</span>
    <h2>Every error the method makes is lift hysteresis</h2>
    <p>Give RC-WFR an <em>oracle</em> lift &mdash; redraw the fiber configuration from
    the exact conditional after each move &mdash; and it sits at 1.0&ndash;1.1&times; the
    estimator floor at every transport rate from
    <span class="mono">&kappa; = 0.03</span> to <span class="mono">8.0</span>, on both
    systems. The WFR flow itself carries no bias. Replace the oracle by the only lift
    that is actually implementable &mdash; carry the configuration across unchanged
    &mdash; and a systematic floor appears that grows with
    <span class="mono">&kappa;</span> and that more compute does not remove. It is also
    independent of how long you relax after each move: it is a steady-state hysteresis,
    not a post-jump transient.</p>
  </div>
  <figure>{img("fig2_lift_bias", "Free-energy error versus transport rate for identity, model-based and oracle lifts on two fiber types")}
  <figcaption><b>A model-based lift only repairs the modes it models.</b> Rescaling the
  fiber coordinate by <span class="mono">&omega;(x)/&omega;(x&prime;)</span> is exact for
  a harmonic fiber and restores RC-WFR to 1.5&times; the floor (left). On a fiber whose
  slow mode is <em>which channel</em> is occupied &mdash; a mode the model does not
  contain &mdash; the same lift makes the error 1.1&ndash;1.7&times; worse than doing
  nothing (right). A lift built from a local model repairs what one already understands
  and can damage what one does not, which is by construction what made the problem
  hard.</figcaption></figure>
</section>

<hr class="floor">
<section>
  <div class="col">
    <span class="sec-no">04 &mdash; Head to head</span>
    <h2>Nineteen arms, one budget</h2>
    <p>Every arm runs the same number of replicas and the same number of steps, so total
    force evaluations match by construction; replica-exchange energy evaluations are
    charged and its inner loop shortened to compensate. All arms share the estimator,
    the initial ensemble and the seed base, so every comparison is paired.
    Hyper-parameters were screened on separate seeds and frozen before these runs
    &mdash; the baselines at least as hard as RC-WFR.</p>
  </div>
  <figure>{img("fig3_arms", "Dot plot of budget-normalized integrated error with bootstrap intervals for every arm on both systems")}
  <figcaption><b>Every arm, both systems, one budget.</b> Bars are 95% bootstrap
  intervals on the paired median over 32 seeds; the dotted line is the measured
  estimator floor. Open markers use oracle information and are upper bounds, not usable
  methods.</figcaption></figure>
  <div class="col">
    <h3>Entropic bottleneck &mdash; unimodal, fast fiber</h3>
  </div>
  {eb_tbl}
  <p class="caption"><sup>*</sup> uses oracle information (the exact conditional law);
  those rows are upper bounds, not usable methods. Percentages are the paired median
  relative change in <span class="mono">I<sub>F</sub></span> against cold-start
  stratified TI, green where the 95% bootstrap CI excludes zero.</p>
  <div class="col">
    <h3>Hidden two-channel fiber &mdash; the correct channel occupancy runs 1 &rarr; 0
    across the domain and the channels interconvert only near a gateway</h3>
  </div>
  {ch_tbl}
  <p class="caption">Compared here against cold-start RE-TI, the strongest baseline that
  uses no oracle information, screened over window count and exchange period.</p>
  <figure>{img("fig4_curves_CHANNEL", "Free-energy error versus force evaluations for the main arms on the hidden-channel system")}
  <figcaption><b>The bias floor is visible as a plateau.</b> The stochastic-step arm
  flattens out while every exact method keeps descending past it; the probability-flow
  arm keeps descending because its transport shuts itself off as the marginal
  flattens.</figcaption></figure>
</section>

<hr class="floor">
<section>
  <div class="col">
    <span class="sec-no">05 &mdash; Scaling</span>
    <h2>One prediction confirmed, one falsified</h2>
    <p>The marginal argument predicts that RC-WFR should gain on ABF as the CV domain
    lengthens, because ABF equilibrates the CV diffusively while W+FR fronts. On a
    periodic torsional landscape with identical local physics at every length, that is
    exactly what happens.</p>
  </div>
  {tors_tbl}
  <p class="caption">Best configuration per family at a fixed 25.6M force-evaluation
  budget; each family free to trade replica count against steps. Stratified TI also
  degrades with <span class="mono">L</span> &mdash; it needs
  <span class="mono">M ~ L</span> windows for fixed resolution &mdash; and RC-WFR closes
  on it steadily without overtaking it in this family.</p>
  <div class="col">
    <p>The opposite prediction &mdash; that RC-WFR should overtake replica exchange as
    the fiber grows, because exchange acceptance decays with system size &mdash; is
    false. Acceptance decays only slowly, while RC-WFR's lift bias is extensive in the
    number of dragged modes, so the gap widens instead of closing.</p>
  </div>
  {ms_tbl}
  <figure>{img("fig5_scaling", "Best integrated error versus CV domain length, and relative error versus spectator fiber degrees of freedom with exchange acceptance")}
  <figcaption><b>Two opposite scalings.</b> Left: RC-WFR crosses ABF between
  <span class="mono">L = 3</span> and <span class="mono">L = 6</span> and pulls away.
  Right: it falls further behind both stratified baselines as the fiber grows, while
  replica-exchange acceptance drops only from 0.975 to 0.814.</figcaption></figure>
</section>

<hr class="floor">
<section>
  <div class="col">
    <span class="sec-no">06 &mdash; What the controls say</span>
    <h2>Reallocation matters; its geometry does not</h2>
    <ul class="find">
      <li><span class="tag y">holds</span><div><b>The W + FR decomposition is
        necessary in both halves.</b><span>Turning Fisher&ndash;Rao off in the best
        variant costs a factor 2.4 on the easy system and 2.5 on the hidden-channel
        system. Wasserstein alone is 5&ndash;6&times; worse than the pair;
        Fisher&ndash;Rao alone never leaves the starting region (coverage 0.07).</span>
      </div></li>
      <li><span class="tag n">rejected</span><div><b>Smooth Fisher&ndash;Rao is not
        needed.</b><span>Plain count balancing ties it three separate times
        (0.03502 vs 0.03612; 0.01504 vs 0.01513; 0.07804 vs 0.06519), while a
        matched-turnover sham that keeps the timing and intensity but destroys the
        direction is 2.3&times; worse. What matters is <em>that</em> population is
        reallocated toward uniform, not the Fisher&ndash;Rao geometry of it. This
        reproduces the earlier ABF/ABP result in a setting with no adaptive bias to be
        redundant with, which makes it a property of the uniform target.</span>
      </div></li>
      <li><span class="tag">equal</span><div><b>A Gaussian-mixture score matches a
        KDE score.</b><span>With enough components the mixture reproduces the KDE arm's
        performance exactly (0.01381 vs 0.01366) at the same wall clock, giving
        analytic <span class="mono">p</span> and
        <span class="mono">&nabla;log p</span> with no grid differentiation &mdash;
        attractive for higher-dimensional CVs. It is not a drop-in, though: it needs its
        own component count and step size, and too few components is catastrophic rather
        than merely inaccurate (at K = 8 the score drove the entire ensemble into the
        walls).</span></div></li>
      <li><span class="tag n">fails</span><div><b>The deterministic flow cannot start
        from a single structure.</b><span>The score of a delta ensemble vanishes at the
        particles, so the ensemble never moves: coverage stayed at 0.02 for every
        <span class="mono">&kappa;</span> and <span class="mono">&theta;</span> tested.
        Exactly the situation the method is sold for. A brief stochastic phase, or any
        spread in the initial ensemble, fixes it.</span></div></li>
    </ul>
  </div>
</section>

<hr class="floor">
<section>
  <div class="col">
    <span class="sec-no">07 &mdash; When to reach for it</span>
    <h2>The regime is narrow but predictable in advance</h2>
  </div>
  <div class="tw"><table class="regime"><thead><tr><th>condition</th>
  <th>RC-WFR</th></tr></thead><tbody>
    <tr><td>CV domain long relative to physical CV diffusion</td>
        <td class="d pos">beats ABF, margin grows with L</td></tr>
    <tr><td>High enthalpic or entropic barrier the bias must learn</td>
        <td class="d pos">beats ABF and SHUS decisively</td></tr>
    <tr><td>Fiber slow mode with a localized switch region</td>
        <td class="d pos">beats cold-start RE-TI ~2&times;</td></tr>
    <tr><td>Fiber has an exact, cheap analytic lift</td>
        <td class="d pos">beats cold-start stratified TI ~1.6&times;</td></tr>
    <tr><td>Easy unimodal fiber, short CV domain</td>
        <td class="d neg">loses to stratified TI and to ABF</td></tr>
    <tr><td>System size grows</td>
        <td class="d neg">loses further; the bias is extensive in fiber modes</td></tr>
    <tr><td>Only one starting structure available</td>
        <td class="d neg">the flow form cannot start at all</td></tr>
  </tbody></table></div>
  <div class="col">
    <div class="claim">
      <p>Reaction-coordinate WFR is a grid-free, continuum alternative to stratified
      thermodynamic integration whose CV transport is unconditional and therefore biased
      by exactly the fiber modes it drags. Its Fisher&ndash;Rao half is hysteresis-free
      and does most of the useful work; its Wasserstein half should be run
      deterministically and gently.</p>
    </div>
    <h3>What would have to be true for it to be general</h3>
    <p>A lift that is asymptotically exact <em>without knowing</em>
    <span class="mono">F</span>. Four kinds were tried and none qualifies: the oracle
    (not implementable); a model-based rescaling (repairs only modelled modes, damages
    unmodelled ones); annealing <span class="mono">&kappa; &rarr; 0</span> (removes the
    bias only by removing the transport, converging to stratified TI); and the
    deterministic probability flow (self-annihilating, the best variant found, but still
    extensive in dragged modes). Any fifth candidate must supply the missing
    <span class="mono">exp(+&beta;F)</span> weight from somewhere other than an estimate
    of <span class="mono">F</span>, and the only known mechanism that does is exchange
    between occupied windows.</p>
    <p>Two directions the campaign did not close. A <em>hybrid</em>: use RC-WFR's front
    to establish coverage, then hand the configurations to exact RE-TI &mdash; the two
    mechanisms are complementary, and the annealed variant is already a crude version of
    it. And <em>variance-optimal targets</em>: everything here targets uniform
    <span class="mono">u(z)</span>, which allocates computation evenly rather than where
    the mean-force variance is. That is an allocation question, entirely separate from
    the bias question, and the Fisher&ndash;Rao machinery carries any target at no extra
    cost.</p>
  </div>
</section>

<footer class="col">
  <p>Every number here is produced by a script in <code>scripts/</code> and recorded, in
  the order it was measured, in <code>docs/RESULTS_LOG.md</code>. No claim is made about
  differences at or below the measured estimator floor &mdash; a check that invalidated
  this campaign's own first result. Wall clock is reported separately: RC-WFR's marginal
  machinery costs about 1.5&times; the wall clock of stratified TI in these toys, where a
  force evaluation is a two-term polynomial; in any real system the force cost dominates
  and that overhead vanishes.</p>
</footer>
</div>
"""
    out = HERE / "report.html"
    out.write_text(head + body)
    print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB)")
    return out


if __name__ == "__main__":
    build()
