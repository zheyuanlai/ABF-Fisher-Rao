"""Emit base64 <img> tags for the campaign figures (the Artifact CSP blocks external
hosts, so every image must be inlined)."""
from __future__ import annotations
import base64, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def img_tag(name, alt, cls="fig"):
    p = HERE / f"{name}.png"
    b = base64.b64encode(p.read_bytes()).decode()
    return f'<img class="{cls}" alt="{alt}" src="data:image/png;base64,{b}">'


if __name__ == "__main__":
    out = {}
    for n, alt in [
        ("fig1_mechanism", "Left: KL divergence to uniform versus time for W-only, "
         "FR-only and W+FR particle runs against the WFR PDE. Right: log-log plot of "
         "time to reach KL below 0.05 against CV domain half-width, showing W scaling "
         "as L squared and W+FR scaling as L."),
        ("fig2_lift_bias", "Free-energy error at full budget versus transport rate "
         "kappa for identity, model-based and oracle lifts, on a harmonic fiber and on "
         "a hidden-channel fiber."),
        ("fig3_arms", "Dot plot of budget-normalized integrated free-energy error with "
         "bootstrap confidence intervals for every method arm on the EB and CHANNEL "
         "systems."),
        ("fig4_curves_EB", "Free-energy error versus force evaluations for the main "
         "arms on the EB system, showing RC-WFR plateauing at a bias floor while the "
         "exact methods continue to descend."),
        ("fig4_curves_CHANNEL", "Free-energy error versus force evaluations for the "
         "main arms on the hidden-channel system."),
        ("fig5_scaling", "Left: best integrated error versus CV domain length for "
         "RC-WFR, ABF, fixed-window TI and RE-TI. Right: relative integrated error "
         "versus number of spectator fiber degrees of freedom, with replica-exchange "
         "acceptance on a second axis."),
    ]:
        if (HERE / f"{n}.png").exists():
            out[n] = img_tag(n, alt)
    (HERE / "_embedded.json").write_text(json.dumps(out))
    print(f"embedded {len(out)} figures, "
          f"{sum(len(v) for v in out.values())/1e6:.2f} MB of base64")
