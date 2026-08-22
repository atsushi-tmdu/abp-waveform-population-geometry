#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WF-P release-quality publication assets
=======================================

Presentation-only renderer for the WF-P GitHub/Zenodo release.

Scientific rules:
- no model fitting;
- no scientific metric recomputation;
- no patient-level data;
- frozen B8 unchanged;
- Release Figure 1 = population-common geometry;
- Release Figure 2 = between/within/replicate scale;
- constitutional / phenotype results remain tables, not forced into figures.

Inputs
------
1) Existing FIG0 aggregate exports:
   - fig2_reconstruction_curves.csv
   - fig2_summary_metrics.csv
2) Aggregate release-candidate tables bundled with this renderer.

Outputs
-------
- Release_Figure_1_population_geometry.{pdf,png,svg}
- Release_Figure_2_between_within_scale.{pdf,png,svg}
- panel files
- release tables copied as CSV
- release tables rendered to Markdown
- captions
- SHA256 manifest

No patient cloud is used.
"""

from __future__ import annotations
import argparse, csv, hashlib, math, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

DPI = 300
PANEL_SIZE = (5.5, 3.9)

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):
            h.update(c)
    return h.hexdigest()

def metric_map(df):
    return {str(r["metric"]): float(r["value"]) for _,r in df.iterrows()}

def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

def panel(letter):
    fig=plt.figure(figsize=PANEL_SIZE)
    fig.text(0.015,0.975,letter,fontsize=15,fontweight="bold",ha="left",va="top")
    ax=fig.add_axes([0.18,0.16,0.76,0.74])
    return fig,ax

def save(fig,base):
    fig.savefig(base.with_suffix(".svg"),bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"),bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"),dpi=DPI,bbox_inches="tight")
    plt.close(fig)

def combine(pngs,out_png):
    ims=[Image.open(p).convert("RGB") for p in pngs]
    gap=40
    w=sum(x.width for x in ims)+gap*(len(ims)-1)
    h=max(x.height for x in ims)
    canvas=Image.new("RGB",(w,h),(255,255,255))
    x=0
    for im in ims:
        y=(h-im.height)//2
        canvas.paste(im,(x,y))
        x += im.width+gap
    canvas.save(out_png,dpi=(DPI,DPI))
    canvas.save(out_png.with_suffix(".pdf"),"PDF",resolution=DPI)

def md_table(df):
    cols=list(df.columns)
    lines=["| "+" | ".join(cols)+" |","| "+" | ".join(["---"]*len(cols))+" |"]
    for _,r in df.iterrows():
        vals=[]
        for c in cols:
            v=r[c]
            if isinstance(v,float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| "+" | ".join(vals)+" |")
    return "\n".join(lines)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--fig0-dir",default="~/Documents/abp_information_study/results/wfp_fig0_exports")
    ap.add_argument("--data-dir",default=None,help="Bundled aggregate table directory; defaults next to this script/data")
    ap.add_argument("--out",default="~/Documents/abp_information_study/results/wfp_release_publication_assets")
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test:
        print("WF-P release-publication renderer self-test: PASS")
        return 0

    fig0=Path(a.fig0_dir).expanduser().resolve()
    data=Path(a.data_dir).expanduser().resolve() if a.data_dir else Path(__file__).resolve().parent/"data"
    out=Path(a.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    (out/"tables").mkdir(exist_ok=True)

    curves=pd.read_csv(fig0/"fig2_reconstruction_curves.csv")
    smdf=pd.read_csv(fig0/"fig2_summary_metrics.csv")
    sm=metric_map(smdf)
    pop=pd.read_csv(data/"Table_WFP_population_geometry.csv")
    scale=pd.read_csv(data/"Table_WFP_scale_interface.csv")
    const=pd.read_csv(data/"Table_WFP_constitutional_and_phenotype_summary.csv")
    pheno=pd.read_csv(data/"Table_WFP_chronic_phenotype_mapping.csv")
    axis=pd.read_csv(data/"Table_WFP_axis_reliability.csv")

    # ---------- Release Figure 1A: reconstruction ----------
    fig,ax=panel("A")
    d=curves["dimension"].to_numpy(float)
    ax.plot(d,curves["replicate_basis_cv_r2_all"],marker="o",markersize=3,
            label="Replicate-stable basis")
    ax.plot(d,curves["ordinary_pca_cv_r2_all"],marker="s",markersize=3,
            linestyle="--",label="Ordinary PCA")
    ax.plot(d,curves["fourier_cv_r2_all"],marker="^",markersize=3,
            linestyle=":",label="Fourier")
    ax.axvline(6,linestyle=":",linewidth=1)
    ax.axvline(8,linestyle="--",linewidth=1)
    ax.text(6.15,0.36,"d90=6",rotation=90,fontsize=8,va="bottom")
    ax.text(8.15,0.36,"d95=8 / frozen B8",rotation=90,fontsize=8,va="bottom")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Held-out reconstruction $R^2$")
#     ax.set_title("Population reconstruction generalizes to held-out patients",fontsize=11)
    ax.legend(frameon=False,fontsize=8,loc="lower right")
    style(ax)
    p1a=out/"Release_Figure_1A_reconstruction"
    save(fig,p1a)

    # ---------- Release Figure 1B: stability ----------
    labels=[
        "Within-window variance\ncaptured by B8",
        "Between/within\nprojector overlap",
        "Half-split\nsubspace overlap",
    ]
    vals=[
        float(pop.loc[pop.metric=="Within-window variance captured by B8","value"].iloc[0]),
        float(pop.loc[pop.metric=="Between/within projector overlap","value"].iloc[0]),
        float(pop.loc[pop.metric=="Half-split subspace overlap median","value"].iloc[0]),
    ]
    fig,ax=panel("B")
    y=np.arange(len(labels))
    ax.scatter(vals,y,s=48)
    for x,yy in zip(vals,y):
        ax.annotate(f"{x:.3f}",(x,yy),xytext=(5,0),textcoords="offset points",
                    ha="left",va="center",fontsize=8.5)
    ax.set_yticks(y,labels)
    ax.set_xlim(0.89,1.005)
    ax.set_xlabel("Fraction / overlap")
#     ax.set_title("The frozen population space is stable across resampling and scale",fontsize=11)
    style(ax)
    p1b=out/"Release_Figure_1B_stability"
    save(fig,p1b)
    combine([p1a.with_suffix(".png"),p1b.with_suffix(".png")],
            out/"Release_Figure_1_population_geometry.png")

    
    # ---------- Release Figure 2A: scale hierarchy ----------
    def get(metric):
        return float(scale.loc[scale.metric==metric,"value"].iloc[0])

    between_rms = get("Between-patient pairwise RMS")
    within_rms = get("Within-patient equal-patient RMS")
    replicate_rms = get("Odd/even replicate RMS")
    adj_step = get("Adjacent 60-s step median")
    nn_median = get("Nearest-neighbor median")
    frac_cross = get("Patients with p95 block displacement >= nearest-neighbor")
    ratio_within_between = get("Within / between")
    ratio_between_replicate = get("Between / replicate")

    fig,ax=panel("A")
    labels = ["Odd/even replicate", "Within-patient 60-s", "Between-patient"]
    vals = [replicate_rms, within_rms, between_rms]
    y = np.arange(len(labels))

    for v, yy in zip(vals, y):
        ax.scatter([v], [yy], s=58, zorder=3)
        ax.annotate(f"{v:.3f}", (v, yy), xytext=(6,0),
                    textcoords="offset points", ha="left", va="center", fontsize=8.5)

    ax.set_yticks(y, labels)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 4.0)
    ax.set_xticks([0.1, 0.2, 0.5, 1.0, 2.0, 4.0])
    ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "4"])
    ax.set_xlabel("RMS distance in frozen B8 (log scale)")
    ax.set_ylabel("")
    ax.text(
        0.98, 0.08,
        f"Within / between = {ratio_within_between:.3f}\n"
        f"Between / replicate = {ratio_between_replicate:.1f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5
    )
    style(ax)
    p2a=out/"Release_Figure_2A_scale_hierarchy"
    save(fig,p2a)

    # ---------- Release Figure 2B: local spacing ----------
    fig,ax=panel("B")
    labels = ["Adjacent 60-s step\nmedian", "Nearest-neighbor\nmedian"]
    vals = [adj_step, nn_median]
    y = np.arange(len(labels))

    for v, yy in zip(vals, y):
        ax.plot([0, v], [yy, yy], linewidth=1.4)
        ax.scatter([v], [yy], s=52, zorder=3)
        ax.annotate(f"{v:.3f}", (v, yy), xytext=(6,0),
                    textcoords="offset points", ha="left", va="center", fontsize=8.5)

    ax.set_yticks(y, labels)
    ax.set_xlim(0.0, max(vals)*1.10)
    ax.set_xlabel("Distance in frozen B8")
    ax.set_ylabel("")
    ax.text(
        0.98, 0.10,
        f"{100*frac_cross:.1f}% of patients had\n"
        f"p95 block displacement ≥\n"
        f"nearest-neighbor distance",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5
    )
    style(ax)
    p2b=out/"Release_Figure_2B_local_spacing"
    save(fig,p2b)

    combine([p2a.with_suffix(".png"),p2b.with_suffix(".png")],
            out/"Release_Figure_2_between_within_scale.png")


    # Copy and render aggregate tables.
    tables={
      "Table_1_population_geometry.csv":pop,
      "Table_2_between_within_scale.csv":scale,
      "Table_3_constitutional_sensitivity.csv":const,
      "Table_4_chronic_phenotype_mapping.csv":pheno,
      "Table_S1_axis_reliability.csv":axis,
    }
    md_parts=["# WF-P release tables\n"]
    for fn,df in tables.items():
        df.to_csv(out/"tables"/fn,index=False)
        md_parts.append("## "+fn.replace(".csv","").replace("_"," ")+"\n")
        md_parts.append(md_table(df)+"\n")
    (out/"WFP_RELEASE_TABLES.md").write_text("\n".join(md_parts),encoding="utf-8")

    captions=f"""# WF-P release figure captions

## Release Figure 1. Stable low-dimensional population-common morphology geometry.

**A.** Held-out reconstruction of 30-min patient central morphology across
dimension using the replicate-stable population basis, ordinary PCA, and a
fixed Fourier basis. The frozen B8 dimension corresponds to d95=8; d90=6 is
also marked. **B.** Stability and between-within alignment summaries for the
frozen B8 space: within-window variance captured by B8, between/within
projector overlap, and median half-split subspace overlap. Panel B is displayed over a restricted x-axis range (0.90–1.00) to
resolve differences among values near unity. This figure characterizes a
reusable population coordinate system and does not assign physiological labels
to individual axes.

## Release Figure 2. Between-patient separation and short-window movement in the frozen B8 space.

**A.** RMS distance scales for odd/even replicate discrepancy, equal-patient
60-s within-patient movement, and between-patient separation. **B.** Median
adjacent 60-s step and median nearest-neighbor patient distance; the annotation
reports the fraction of patients whose 95th-percentile block displacement was
at least as large as their nearest-neighbor distance. Sixty-second movement is
not interpreted as a long-duration WF3 trajectory.
"""
    (out/"WFP_RELEASE_FIGURE_CAPTIONS.md").write_text(captions,encoding="utf-8")

    boundary="""# Release presentation boundary

- The patient z1-z2 cloud is intentionally omitted from the public release.
- Constitutional and chronic-phenotype results are retained as tables rather than forced into a main figure.
- The frozen B8 interface is the primary release object.
- Release figures are reusable integration assets for a future WF1/WF2/WF-P/WF3 manuscript.
- No B8 axis is labeled as trait, state, disease, or treatment-response direction.
- Weak constitutional conditional-mean predictability does not imply independence.
"""
    (out/"WFP_RELEASE_PRESENTATION_BOUNDARY.md").write_text(boundary,encoding="utf-8")

    rows=[]
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name!="WFP_RELEASE_ASSET_MANIFEST_SHA256.csv":
            rows.append({"file":str(p.relative_to(out)),"bytes":p.stat().st_size,"sha256":sha256_file(p)})
    with (out/"WFP_RELEASE_ASSET_MANIFEST_SHA256.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["file","bytes","sha256"]); w.writeheader(); w.writerows(rows)

    txt="\n".join([
      "WF-P RELEASE PUBLICATION ASSETS",
      "===============================",
      "Decision: WFP_RELEASE_PUBLICATION_ASSETS_COMPLETE",
      "Scientific effects calculated: NO",
      "Patient-level data used: NO",
      "Frozen B8 changed: NO",
      "Public release figures: 2",
      "Public release tables: 5",
      "Patient cloud included: NO",
      "Constitutional figure included: NO (table retained)",
      "",
      "Next step:",
      "  Visual-QC the two release figures.",
      "  If accepted, build the separate GitHub staging tree.",
      "",
    ])
    (out/"WFP_RELEASE_PUBLICATION_ASSETS_READOUT.txt").write_text(txt,encoding="utf-8")
    print(txt)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
