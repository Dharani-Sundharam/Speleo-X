"""
=============================================================================
SPELEO-X  ·  SUBTERRANEAN SENSING ENGINE  —  PROTOTYPE V2
=============================================================================
Processes any RGB image of cave walls or crop fields to produce:
  • Phase 1 — Pseudo-NIR channel + NIR-R-G False Colour Composite
  • Phase 2 — IR-threshold Mineral Classification Heatmap
  • Phase 3 — LiDAR-proxy Structural Geometry Density Map
  • Phase 4 — CNN-based Mineral Identification (MobileNetV2, 7 classes)

Output : Research-paper figure PNG  (dark background, clean subfigure layout)

Usage
-----
  python spectral_pipeline.py <image.jpg>
  python spectral_pipeline.py <image.jpg> output.png
=============================================================================
"""

import sys, os, json, textwrap
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import gaussian_filter

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & THEME
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "mineral_classifier.h5")
IDX_PATH    = os.path.join(BASE_DIR, "class_indices.json")

BG          = "#080810"
ACCENT      = "#00FFD1"
ACCENT2     = "#7B61FF"
GOLD        = "#FFD166"
RED         = "#E63946"
TEXT_DIM    = "#5A6070"
TEXT_MID    = "#9CA3AF"
TEXT_BRIGHT = "#E8EAED"
BORDER      = "#1E2235"

MINERAL_PALETTE = {
    "biotite"    : "#4A90D9",
    "bornite"    : "#D4A017",
    "chrysocolla": "#3CB371",
    "malachite"  : "#2ECC71",
    "muscovite"  : "#C8B89A",
    "pyrite"     : "#FFD700",
    "quartz"     : "#A8E6FF",
}

GEO_INFO = {
    "biotite"    : ("K(Mg,Fe)₃AlSi₃O₁₀(OH)₂", "Monoclinic",
                    "Igneous & metamorphic rocks, granites, schists",
                    MINERAL_PALETTE["biotite"]),
    "bornite"    : ("Cu₅FeS₄",                  "Orthorhombic",
                    "Copper ore deposits, hydrothermal veins",
                    MINERAL_PALETTE["bornite"]),
    "chrysocolla": ("(Cu,Al)₂H₂Si₂O₅(OH)₄·nH₂O","Amorphous",
                    "Oxidised zones of copper deposits, cave walls",
                    MINERAL_PALETTE["chrysocolla"]),
    "malachite"  : ("Cu₂(CO₃)(OH)₂",            "Monoclinic",
                    "Weathered copper ore, cave formations",
                    MINERAL_PALETTE["malachite"]),
    "muscovite"  : ("KAl₂(AlSi₃O₁₀)(OH)₂",     "Monoclinic",
                    "Granite, schist, pegmatites",
                    MINERAL_PALETTE["muscovite"]),
    "pyrite"     : ("FeS₂",                      "Cubic",
                    "Hydrothermal veins, sedimentary/igneous contact zones",
                    MINERAL_PALETTE["pyrite"]),
    "quartz"     : ("SiO₂",                      "Trigonal",
                    "Ubiquitous in continental crust, cave speleothems",
                    MINERAL_PALETTE["quartz"]),
}

THRESH_HIGH    = 210
THRESH_MID_LOW = 130
IR_W_R, IR_W_G, IR_W_B = 0.70, 0.25, 0.05
CANNY_T1, CANNY_T2     = 50, 150


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — SPECTRAL RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_pseudo_ir(bgr: np.ndarray) -> np.ndarray:
    """
    Estimate a Pseudo Near-Infrared channel from RGB.

    Geological Logic
    ----------------
    Minerals such as Calcite absorb strongly in the blue band;
    Gypsum/Quartz show elevated NIR-proxy reflectance in red.
    We weight R heavily and suppress B to simulate NIR response.
    """
    b, g, r = cv2.split(bgr.astype(np.float32))
    return np.clip(IR_W_R * r + IR_W_G * g + IR_W_B * b, 0, 255)


def build_false_colour_composite(bgr: np.ndarray,
                                  ir: np.ndarray) -> np.ndarray:
    """
    Construct a NIR-R-G False Colour Composite.

    Geological Logic
    ----------------
    Standard RS false-colour convention maps (NIR→R, Red→G, Green→B).
    Hydrated mineral crusts (moonmilk, flowstones) appear in vivid
    warm tones that are invisible in the true-colour image.
    """
    _, g, r = cv2.split(bgr)
    return cv2.merge([g, r, ir.astype(np.uint8)])


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — MINERAL HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def classify_minerals(ir: np.ndarray) -> np.ndarray:
    """
    Threshold-based three-class spectral classification.

    Geological Logic
    ----------------
    DN thresholds derived from known mineral end-member spectra:
        Type A  (DN > 210)       — Gypsum / Quartz     (high reflectors)
        Type B  (130 < DN ≤ 210) — Calcite / Limestone (intermediate)
        Type C  (DN ≤ 130)       — Base Rock / Shadow  (low reflectors)
    """
    m = np.zeros(ir.shape, dtype=np.uint8)
    m[ir > THRESH_HIGH]                                  = 255
    m[(ir > THRESH_MID_LOW) & (ir <= THRESH_HIGH)]       = 128
    return m


def apply_mineral_heatmap(mm: np.ndarray) -> np.ndarray:
    """Apply COLORMAP_HOT with mild smoothing to reduce classification noise."""
    return cv2.applyColorMap(cv2.GaussianBlur(mm, (5, 5), 0), cv2.COLORMAP_HOT)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — STRUCTURAL GEOMETRY / LiDAR PROXY
# ─────────────────────────────────────────────────────────────────────────────

def compute_structural_geometry(bgr: np.ndarray) -> np.ndarray:
    """
    Simulate Livox Mid-360 point-cloud density.

    Geological Logic
    ----------------
    LiDAR point density peaks at geometric discontinuities (cave wall
    junctions, stalactite bases, ledge overhangs).  We approximate this
    by inverting the Distance Transform of Canny edges:  pixels near
    edges = structurally complex = high simulated density.
    Gaussian blur models sensor beam divergence.
    """
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), CANNY_T1, CANNY_T2)
    dist  = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 5)
    inv   = dist.max() - dist
    smo   = gaussian_filter(inv, sigma=4)
    lo, hi = smo.min(), smo.max()
    return (smo - lo) / (hi - lo + 1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — CNN MINERAL IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def load_classifier():
    """
    Load the saved MobileNetV2 mineral classifier.
    Returns (model, idx_to_class) or (None, None) if not found.
    """
    if not os.path.isfile(MODEL_PATH):
        print(f"[!] Model not found: {MODEL_PATH}")
        print("    → Run:  python train_mineral_classifier.py\n")
        return None, None
    import tensorflow as tf
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(IDX_PATH) as f:
        c2i = json.load(f)
    idx_to_class = {v: k for k, v in c2i.items()}
    print(f"[✓] Classifier loaded  ({len(idx_to_class)} classes)")
    return model, idx_to_class


def predict_minerals(bgr: np.ndarray, model, idx_to_class: dict,
                     top_k: int = 3):
    """
    Run whole-image inference and return top-k (class, confidence) pairs.

    Geological Logic
    ----------------
    MobileNetV2 was fine-tuned on 7-class hand-specimen mineral images.
    Softmax probabilities reflect likelihood of each mineral type being
    the dominant constituent of the visible rock surface.
    """
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img   = cv2.resize(rgb, (160, 160)).astype(np.float32) / 255.0
    probs = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    top   = np.argsort(probs)[::-1][:top_k]
    return [(idx_to_class[i], float(probs[i])) for i in top]


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD — HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _theme():
    matplotlib.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor"  : BG,
        "axes.edgecolor"  : BORDER,
        "text.color"      : TEXT_BRIGHT,
        "font.family"     : "monospace",
        "axes.grid"       : False,
    })


def _border(ax, color=BORDER, lw=0.8):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(color); sp.set_linewidth(lw)


def _label(ax, letter, title, lc=ACCENT, tc=TEXT_DIM):
    """Bottom-left subfigure label  — research paper style."""
    ax.text(0.013, 0.03, f"({letter})",
            transform=ax.transAxes, color=lc, fontsize=9,
            fontweight="bold", va="bottom", fontfamily="monospace",
            bbox=dict(facecolor="#00000099", edgecolor="none",
                      boxstyle="round,pad=0.18"))
    ax.text(0.075, 0.03, title,
            transform=ax.transAxes, color=tc, fontsize=7,
            va="bottom", fontfamily="monospace",
            bbox=dict(facecolor="#00000099", edgecolor="none",
                      boxstyle="round,pad=0.18"))


def _cbar(fig, ax, mappable, label):
    cb = fig.colorbar(mappable, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(label, color=TEXT_DIM, fontsize=6, labelpad=3)
    cb.ax.tick_params(labelsize=5.5, color=TEXT_DIM, length=2)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_DIM)
    cb.outline.set_edgecolor(BORDER); cb.outline.set_linewidth(0.5)
    return cb


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD — MINERAL REPORT PANEL
# ─────────────────────────────────────────────────────────────────────────────

def _mineral_report(ax, predictions, img_shape):
    """
    Full-width bottom panel: confidence bars + geology card + sensor metadata.
    Three columns separated by thin vertical rules.
    """
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_facecolor("#0C0C18")
    # top accent line
    ax.plot([0, 1], [0.99, 0.99], color=ACCENT2, lw=0.9,
            transform=ax.transAxes, clip_on=False)

    # ── Column header ─────────────────────────────────────────────────────
    ax.text(0.0, 0.96,
            "MINERAL IDENTIFICATION  ·  CNN Inference  [MobileNetV2 / Transfer Learning]",
            color=ACCENT, fontsize=7.8, fontweight="bold",
            va="top", fontfamily="monospace")

    if predictions is None:
        ax.text(0.32, 0.48,
                "⚠  Model not found — run train_mineral_classifier.py",
                color=GOLD, fontsize=9, ha="center", va="center",
                fontfamily="monospace")
        return

    top_name, top_conf = predictions[0]

    # ── Col A: Confidence bars  (0.0 – 0.35) ─────────────────────────────
    BAR_MAX = 0.28
    BAR_H   = 0.145
    Y_START = 0.73
    ROW_GAP = 0.23

    for i, (name, conf) in enumerate(predictions[:3]):
        col = MINERAL_PALETTE.get(name, ACCENT)
        y   = Y_START - i * ROW_GAP
        lbl = ["1st  Match", "2nd  Match", "3rd  Match"][i]

        ax.text(0.0, y + BAR_H + 0.03, lbl,
                color=TEXT_DIM, fontsize=5.5, va="bottom",
                fontfamily="monospace")

        # track
        ax.add_patch(FancyBboxPatch(
            (0.0, y), BAR_MAX, BAR_H,
            boxstyle="square,pad=0",
            facecolor="#0F0F22", edgecolor=BORDER, lw=0.4))
        # fill
        ax.add_patch(FancyBboxPatch(
            (0.0, y), BAR_MAX * conf, BAR_H,
            boxstyle="square,pad=0",
            facecolor=col, edgecolor="none", alpha=0.9))
        # mineral name inside bar
        ax.text(0.006, y + BAR_H / 2,
                name.capitalize(),
                color="#080810" if conf > 0.28 else col,
                fontsize=7, fontweight="bold",
                va="center", fontfamily="monospace")
        # confidence %
        ax.text(BAR_MAX + 0.015, y + BAR_H / 2,
                f"{conf * 100:.1f}%",
                color=col, fontsize=9.5, fontweight="bold",
                va="center", fontfamily="monospace")

    # vertical rule
    ax.plot([0.38, 0.38], [0.06, 0.94], color=BORDER, lw=0.7)

    # ── Col B: Geology card  (0.40 – 0.72) ───────────────────────────────
    info = GEO_INFO.get(top_name)
    GX   = 0.41

    if info:
        formula, crystal, occurrence, swatch = info

        # swatch strip
        ax.add_patch(FancyBboxPatch(
            (GX, 0.72), 0.013, 0.215,
            boxstyle="square,pad=0",
            facecolor=swatch, edgecolor="none"))

        # name + confidence
        ax.text(GX + 0.024, 0.845,
                top_name.upper(),
                color=TEXT_BRIGHT, fontsize=12.5, fontweight="bold",
                va="center", fontfamily="monospace")
        ax.text(GX + 0.024, 0.745,
                f"Confidence  {top_conf * 100:.1f}%",
                color=swatch, fontsize=7.5, fontweight="bold",
                va="center", fontfamily="monospace")

        # property rows
        rows = [
            ("Formula",      formula),
            ("Crystal Sys.", crystal),
            ("Occurrence",   textwrap.shorten(occurrence, 58, placeholder="…")),
        ]
        for j, (lbl, val) in enumerate(rows):
            ry = 0.57 - j * 0.165
            ax.text(GX,         ry, lbl, color=TEXT_DIM, fontsize=6,
                    va="top", fontfamily="monospace")
            ax.text(GX + 0.11,  ry, val, color=TEXT_MID, fontsize=6.4,
                    va="top", fontfamily="monospace")
            if j < 2:
                ax.plot([GX, 0.72], [ry - 0.042, ry - 0.042],
                        color=BORDER, lw=0.35, alpha=0.7)

    # vertical rule
    ax.plot([0.735, 0.735], [0.06, 0.94], color=BORDER, lw=0.7)

    # ── Col C: Sensor / model metadata  (0.75 – 1.0) ─────────────────────
    h_px, w_px = img_shape[:2]
    meta = [
        ("Input Resolution", f"{w_px} × {h_px} px"),
        ("Sensor (RGB)",     "OAK-D Pro PoE"),
        ("LiDAR Sensor",     "Livox Mid-360"),
        ("CNN Backbone",     "MobileNetV2"),
        ("Pre-training",     "ImageNet"),
        ("Fine-tune Classes","7 mineral types"),
        ("Val. Accuracy",    "77.2 %"),
    ]
    MX  = 0.75
    MY  = 0.95
    DY  = 0.126

    for k, (lbl, val) in enumerate(meta):
        ry = MY - k * DY
        ax.text(MX,   ry, lbl, color=TEXT_DIM, fontsize=5.8,
                va="top", fontfamily="monospace")
        ax.text(0.995, ry, val, color=ACCENT2, fontsize=6,
                fontweight="bold", ha="right", va="top",
                fontfamily="monospace")
        if k < len(meta) - 1:
            ax.plot([MX, 0.995], [ry - 0.052, ry - 0.052],
                    color=BORDER, lw=0.3, alpha=0.55)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def render_dashboard(bgr        : np.ndarray,
                     pseudo_ir  : np.ndarray,
                     fcc        : np.ndarray,
                     heatmap    : np.ndarray,
                     mineral_map: np.ndarray,
                     density    : np.ndarray,
                     predictions,
                     output_path: str = "speleo_dashboard.png") -> None:
    """
    Compose all pipeline outputs into a publication-ready figure.

    Layout
    ──────
      ┌───────────────┬───────────────┐
      │ (a) Visible   │ (b) IR Scan   │   ← equal height, equal width
      ├───────────────┼───────────────┤
      │ (c) Heatmap   │(d) LiDAR Dens │   ← equal height, equal width
      ├───────────────┴───────────────┤
      │   Mineral ID Report (full)    │   ← clean 3-column info panel
      └───────────────────────────────┘
    """
    _theme()

    fig = plt.figure(figsize=(18, 13), facecolor=BG)

    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        left=0.03, right=0.97,
        top=0.975, bottom=0.028,
        hspace=0.055,
        height_ratios=[1, 1, 0.45],
    )
    row0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.035)
    row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.035)

    # ── (a) Visible Spectrum ──────────────────────────────────────────────
    ax_a = fig.add_subplot(row0[0])
    ax_a.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ax_a.set_xticks([]); ax_a.set_yticks([])
    _border(ax_a)
    _label(ax_a, "a", "Visible Spectrum")

    # ── (b) Simulated NIR Reflectance ─────────────────────────────────────
    ax_b = fig.add_subplot(row0[1])
    ax_b.imshow(cv2.cvtColor(fcc, cv2.COLOR_BGR2RGB))
    ax_b.set_xticks([]); ax_b.set_yticks([])
    _border(ax_b, color=ACCENT2)
    _label(ax_b, "b", "Simulated NIR Reflectance  [NIR–R–G composite]",
           lc=ACCENT2)
    sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 255))
    sm.set_array([])
    _cbar(fig, ax_b, sm, "IR Proxy  (DN)")

    # ── (c) Mineral Classification ────────────────────────────────────────
    ax_c = fig.add_subplot(row1[0])
    ax_c.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    ax_c.set_xticks([]); ax_c.set_yticks([])
    _border(ax_c, color=GOLD)
    _label(ax_c, "c", "Mineral Classification Heatmap  [IR threshold]", lc=GOLD)
    pa = mpatches.Patch(facecolor="#FFFFFF", edgecolor="none",
                        label="Gypsum / Quartz   (IR > 210)")
    pb = mpatches.Patch(facecolor="#FF6611", edgecolor="none",
                        label="Calcite / Limestone   (130 – 210)")
    pc = mpatches.Patch(facecolor="#2A0400", edgecolor=BORDER,
                        label="Base Rock   (< 130)")
    ax_c.legend(handles=[pa, pb, pc], loc="upper right",
                fontsize=6, framealpha=0.78,
                facecolor="#0A0A14", edgecolor=GOLD,
                labelcolor=TEXT_BRIGHT, handlelength=1.1,
                borderpad=0.55, labelspacing=0.45)

    # ── (d) Structural Geometry / LiDAR Density ───────────────────────────
    ax_d = fig.add_subplot(row1[1])
    im_d = ax_d.imshow(density, cmap="plasma", vmin=0, vmax=1)
    ax_d.set_xticks([]); ax_d.set_yticks([])
    _border(ax_d, color=RED)
    _label(ax_d, "d", "Simulated LiDAR Point Density  [Livox Mid-360 proxy]", lc=RED)
    _cbar(fig, ax_d, im_d, "Point Density  (normalised)")

    # ── Mineral ID Report ─────────────────────────────────────────────────
    ax_r = fig.add_subplot(outer[2])
    _mineral_report(ax_r, predictions, bgr.shape)

    # ── Minimal footer ────────────────────────────────────────────────────
    fig.text(0.03, 0.005,
             "Speleo-X  ·  Geological Digital Twin Engine  ·  v2.0.0-α",
             ha="left", color=TEXT_DIM, fontsize=5.5, fontfamily="monospace")
    fig.text(0.97, 0.005,
             "CONFIDENTIAL  ·  Livox Mid-360  ·  OAK-D Pro PoE  ·  MobileNetV2",
             ha="right", color=TEXT_DIM, fontsize=5.5, fontfamily="monospace")

    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print(f"\n[✓] Dashboard saved → {os.path.abspath(output_path)}\n")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(image_path: str,
                 output_path: str = "speleo_dashboard.png") -> None:
    """Execute all four phases and render the publication-ready dashboard."""

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"OpenCV could not decode: {image_path}")

    h, w = bgr.shape[:2]
    if max(h, w) > 1200:
        s = 1200 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)),
                         interpolation=cv2.INTER_AREA)

    print(f"\n{'='*58}")
    print(f"  SPELEO-X v2  ·  {os.path.basename(image_path)}")
    print(f"  Resolution   : {bgr.shape[1]} × {bgr.shape[0]} px")
    print(f"{'='*58}\n")

    print("[1/5] Phase 1 — Spectral Reconstruction …")
    pseudo_ir = compute_pseudo_ir(bgr)
    fcc       = build_false_colour_composite(bgr, pseudo_ir)

    print("[2/5] Phase 2 — Mineral Classification Heatmap …")
    mineral_map = classify_minerals(pseudo_ir)
    heatmap     = apply_mineral_heatmap(mineral_map)

    print("[3/5] Phase 3 — Structural Geometry / LiDAR density …")
    density = compute_structural_geometry(bgr)

    print("[4/5] Phase 4 — CNN Mineral Identification …")
    model, idx_to_class = load_classifier()
    predictions = None
    if model is not None:
        predictions = predict_minerals(bgr, model, idx_to_class)
        print("  Top-3 predictions:")
        for name, conf in predictions:
            bar = "█" * int(conf * 28) + "░" * (28 - int(conf * 28))
            print(f"      {name:<14} {bar}  {conf*100:.1f}%")

    print("\n[5/5] Rendering dashboard …")
    render_dashboard(bgr, pseudo_ir, fcc, heatmap,
                     mineral_map, density, predictions, output_path)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python spectral_pipeline.py <image.jpg>")
        print("  python spectral_pipeline.py <image.jpg> output.png\n")
        print("  (Train model first if not done:  python train_mineral_classifier.py)\n")
        sys.exit(1)

    input_img  = sys.argv[1]
    output_png = sys.argv[2] if len(sys.argv) > 2 else "speleo_dashboard.png"
    run_pipeline(input_img, output_png)
