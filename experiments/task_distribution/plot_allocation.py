"""Standalone plotting: all visual parameters are in STYLE in this one file.

python plot_allocation.py
python plot_allocation.py --data output/plot_data.csv --out figures

Requires Matplotlib only. Does not import or run the simulator.
"""
from pathlib import Path
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


# ===================== EDIT ALL PLOT SETTINGS HERE =====================
STYLE = {
    "figure_size_inches": (3.55, 3.05),
    "dpi": 600,
    "margins": dict(left=.17, right=.98, bottom=.16, top=.97),
    "font_serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "font_size": 8,
    "axis_label_size": 9,
    "tick_label_size": 8,
    "axes_linewidth": .7,
    "policies": ["L3GSA", "Dyncast-task", "CoAP-static"],
    "colors": ["#5F8EC6", "#D97070", "#72AF84"],
    "markers": ["o", "s", "^"],
    "line_width": 1.6,
    "marker_size": 4.3,
    "marker_edge_width": .7,
    "reference_color": "#E6F2FA",
    "reference_bar_width": .72,
    "reference_label": "Ideal proportion",
    "x_label": "Precise task arrival rate (rps)",
    "y_label": "Precise task proportion in site 2 (%)",
    "y_limits": (0, 104),
    "y_tick_interval": 20,
    "x_padding": .48,
    "grid_color": "#C8C8C8",
    "grid_linestyle": ":",
    "grid_linewidth": .45,
    "legend_location": "upper left",
    "legend_font_size": 6.7,
    "legend_border_width": .6,
    "errorbar_line_width": .9,
    "errorbar_capsize": 3.,
    "errorbar_cap_width": .9,
    "output_stem": "allocation_ratio",
    "output_formats": ("png", "pdf"),
}
# ======================================================================

BASE = Path(__file__).resolve().parent


def load_data(path, policies):
    with Path(path).open(newline="",encoding="utf-8-sig") as f:
        reader=csv.DictReader(f)
        cols=["precise_task_rate_rps","ideal_proportion_pct"]+[
            f"{p}_{s}" for p in policies for s in ("mean","min","max")]
        if not reader.fieldnames or not set(cols)<=set(reader.fieldnames):
            raise ValueError("Missing required plot CSV columns")
        rows=[{k:float(r[k]) for k in cols} for r in reader]
    rows.sort(key=lambda r:r["precise_task_rate_rps"])
    if not rows or len({r["precise_task_rate_rps"] for r in rows})!=len(rows):
        raise ValueError("Empty or duplicated x positions")
    for r in rows:
        if r["precise_task_rate_rps"]<0 or not 0<=r["ideal_proportion_pct"]<=100:
            raise ValueError("Invalid x/reference value")
        for p in policies:
            if not 0<=r[p+"_min"]<=r[p+"_mean"]<=r[p+"_max"]<=100:
                raise ValueError("Invalid min/mean/max")
    return rows


def plot(rows,out,style=STYLE):
    s=style
    plt.rcParams.update({"font.family":"serif","font.serif":s["font_serif"],
        "font.size":s["font_size"],"axes.labelsize":s["axis_label_size"],
        "xtick.labelsize":s["tick_label_size"],"ytick.labelsize":s["tick_label_size"],
        "axes.linewidth":s["axes_linewidth"],"pdf.fonttype":42,"ps.fonttype":42})
    fig,ax=plt.subplots(figsize=s["figure_size_inches"])
    fig.subplots_adjust(**s["margins"])
    x=list(range(len(rows)))  # User-requested equally spaced category positions.
    ax.bar(x,[r["ideal_proportion_pct"] for r in rows],width=s["reference_bar_width"],
        color=s["reference_color"],edgecolor="none",linewidth=0,zorder=0)
    handles=[]
    for p,color,marker in zip(s["policies"],s["colors"],s["markers"]):
        means=[r[p+"_mean"] for r in rows]
        errors=[[r[p+"_mean"]-r[p+"_min"] for r in rows],
                [r[p+"_max"]-r[p+"_mean"] for r in rows]]
        a=ax.errorbar(x,means,yerr=errors,color=color,marker=marker,
            linewidth=s["line_width"],markersize=s["marker_size"],
            markerfacecolor=color,markeredgecolor=color,markeredgewidth=s["marker_edge_width"],
            elinewidth=s["errorbar_line_width"],capsize=s["errorbar_capsize"],
            capthick=s["errorbar_cap_width"],clip_on=False,zorder=3)
        a.lines[0].set_label(p);handles.append(a.lines[0])
    handles.append(Patch(facecolor=s["reference_color"],edgecolor="none",label=s["reference_label"]))
    ax.set_xticks(x,[f"{r['precise_task_rate_rps']:g}" for r in rows])
    ax.set_xlabel(s["x_label"],labelpad=3)
    ax.set_ylabel(s["y_label"],labelpad=3)
    ax.set_ylim(*s["y_limits"])
    ax.set_xlim(-s["x_padding"],len(rows)-1+s["x_padding"])
    ax.yaxis.set_major_locator(MultipleLocator(s["y_tick_interval"]))
    ax.set_axisbelow(False)
    ax.grid(axis="y",linestyle=s["grid_linestyle"],color=s["grid_color"],
        linewidth=s["grid_linewidth"],zorder=1)
    ax.tick_params(direction="in",length=2.5,width=.6)
    lg=ax.legend(handles=handles,loc=s["legend_location"],fontsize=s["legend_font_size"],
        frameon=True,fancybox=False,framealpha=1,facecolor="white",edgecolor="black",
        handlelength=1.8,handletextpad=.5,borderpad=.5,labelspacing=.4,borderaxespad=.7)
    lg.get_frame().set_linewidth(s["legend_border_width"])
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for ext in s["output_formats"]:
        target=out/(s["output_stem"]+"."+ext)
        fig.savefig(target,dpi=s["dpi"])
        print(target.resolve())
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=BASE/"output"/"plot_data.csv")
    parser.add_argument("--out",type=Path,default=BASE/"output")
    args=parser.parse_args()
    plot(load_data(args.data,STYLE["policies"]),args.out)


if __name__=="__main__":
    main()
