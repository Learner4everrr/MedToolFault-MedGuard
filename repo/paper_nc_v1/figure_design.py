"""Editorial figure design over frozen evidence, without fitting or regrading.

Nature main-journal references inform hierarchy and comparison, not copied art.
All exported figures use a fixed 183-mm canvas, editable text and vector marks.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch
import numpy as np

INK = "#252B31"
MUTED = "#616B74"
GREY = "#89939D"
BLUE = "#356FA3"
AMBER = "#AF8132"
PURPLE = "#8675A5"
RISK = "#B65D43"
PALE = "#EAF1F7"
FAINT = "#F5F7F9"
GRID = "#E2E7EB"
OTHER = "#D8DDE2"
SUCCESS = LinearSegmentedColormap.from_list("evidence_blue", ["#F5F8FB", "#B3CDE1", "#356FA3"])
HARM = LinearSegmentedColormap.from_list("unsafe_amber", ["#FCF7F2", "#E4BA9E", "#A95036"])
CHANGE = LinearSegmentedColormap.from_list("effect_diverging", ["#B65D43", "#FAFAF8", "#356FA3"])
COST = CHANGE.reversed()
POLICIES = {
    "vanilla": ("Vanilla", GREY, "o"),
    "informed": ("Informed", AMBER, "^"),
    "evidence_only": ("Evidence only", PURPLE, "s"),
    "medguard": ("MedGuard", BLUE, "D"),
    "retry_only": ("Retry only", AMBER, "^"),
}
SHORT = {
    "qwen35_9b": "Qwen3.5", "ministral3_8b": "Ministral", "llama31_8b": "Llama3.1",
    "gptoss20b": "GPT-OSS", "qwen3_8b": "Qwen3 (NT)", "qwen3_8b_thinking": "Qwen3 (T)",
    "qwen25_7b": "Qwen2.5-7B", "qwen25_14b": "Qwen2.5-14B", "gemma3_12b": "Gemma3",
    "phi4_mini": "Phi4-mini",
}
MODELS = ("qwen35_9b", "ministral3_8b", "llama31_8b", "gptoss20b")
TIMES_FAMILIES = ("Times New Roman", "Times", "Nimbus Roman", "Liberation Serif")
HAZARDS = ("interface_drift", "transient_503", "stale", "patient_mismatch", "provenance_conflict", "commit_timeout", "no_commit_timeout")
HAZARD_LABELS = ("Interface\ndrift", "Transient\n503", "Stale\nevidence", "Patient\nmismatch", "Provenance\nconflict", "Committed\ntimeout", "No-commit\ntimeout")


def configure():
    # Nimbus Roman is the installed Times-compatible family, including all styles.
    for name in ("Regular", "Bold", "Italic", "BoldItalic"):
        p = Path(f"/usr/share/fonts/urw-base35/NimbusRoman-{name}.otf")
        if p.exists():
            font_manager.fontManager.addfont(str(p))
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((name for name in TIMES_FAMILIES if name in available), None)
    if font is None:
        raise RuntimeError("Install a Times-family font (Times New Roman, Times, Nimbus Roman or Liberation Serif).")
    plt.rcParams.update({
        "font.family": font, "font.size": 7.2, "text.color": INK,
        "font.serif": [font], "mathtext.fontset": "custom",
        "mathtext.rm": font, "mathtext.it": font+":italic",
        "mathtext.bf": font+":bold", "mathtext.sf": font,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
        "axes.labelsize": 7.2, "xtick.labelsize": 6.8, "ytick.labelsize": 7.1,
        "legend.fontsize": 6.6, "axes.linewidth": .55, "axes.edgecolor": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": .55, "ytick.major.width": .55,
        "xtick.major.size": 2.5, "ytick.major.size": 0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "lines.linewidth": .8, "hatch.linewidth": .35,
    })


def canvas(height=166):
    return plt.figure(figsize=(183 / 25.4, height / 25.4))


def panel(fig, rect, letter, heading, subtitle=None):
    ax = fig.add_axes(rect)
    x, y, w, h = rect
    heading_y = y + h + .22 / fig.get_figheight()
    subtitle_y = y + h + .05 / fig.get_figheight()
    fig.text(x - .029, heading_y, letter, fontsize=10, weight="bold", va="bottom")
    fig.text(x, heading_y, heading, fontsize=8.0, va="bottom")
    if subtitle:
        fig.text(x, subtitle_y, subtitle, fontsize=6.6, color=MUTED, va="bottom")
    return ax


def off(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")


def key(label, color, marker="o", hollow=False):
    return Line2D([], [], color=color, marker=marker, markersize=3.8,
                  markerfacecolor="white" if hollow else color, markeredgewidth=.8,
                  linewidth=0, label=label)


def policy_key(p):
    label, color, marker = POLICIES[p]
    return key(label, color, marker, p == "vanilla")


def legend(fig, handles, x, y, ncol=None):
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(x, y),
               ncol=ncol or len(handles), frameon=False, handletextpad=.4,
               columnspacing=1.2, borderaxespad=0, handlelength=1.0)


def rows(ax, names, xmax=None, xmin=0, xlabel=None, ticks=None, bands=True):
    ax.set_yticks(range(len(names)), names)
    ax.set_ylim(len(names)-.55, -.55)
    if xmax is not None:
        ax.set_xlim(xmin, xmax)
    if ticks is not None:
        ax.set_xticks(ticks)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=4)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", pad=5)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, lw=.45)
    if bands:
        for i in range(0, len(names), 2):
            ax.axhspan(i-.5, i+.5, color=FAINT, zorder=-2, lw=0)


def effect(result, reverse=False):
    if "task_weighted_gain" in result:
        v, (lo, hi) = result["task_weighted_gain"], result["patient_cluster_ci95"]
    else:
        v, (lo, hi) = result["difference"], result["ci95"]
    return (-100*v, -100*hi, -100*lo) if reverse else (100*v, 100*lo, 100*hi)


def dot(ax, y, result, color=BLUE, marker="D", reverse=False, hollow=False):
    v, lo, hi = effect(result, reverse)
    xmin, xmax = ax.get_xlim()
    assert xmin <= lo <= v <= hi <= xmax, "Forest axis would clip an interval"
    ax.errorbar(v, y, xerr=[[v-lo], [hi-v]], fmt=marker, ms=3.7,
                mfc="white" if hollow else color, mec=color, mew=.75,
                color=color, capsize=1.7, capthick=.65, elinewidth=.8, zorder=4)


def forest(ax, names, limits=(-10, 60), ticks=None):
    rows(ax, names, limits[1], limits[0], "Paired difference (percentage points)", ticks)
    ax.axvline(0, color=MUTED, lw=.6, linestyle=(0, (3, 3)), zorder=1)


def luminance(rgb):
    c = np.asarray(rgb[:3])
    c = np.where(c <= .04045, c/12.92, ((c+.055)/1.055)**2.4)
    return float(c @ np.array([.2126, .7152, .0722]))


def contrast_text(rgba):
    # Select true black or white by contrast, rather than an arbitrary value cutoff.
    lum = luminance(rgba)
    return "#000000" if (lum+.05)/.05 >= 1.05/(lum+.05) else "#FFFFFF"


def heat(ax, values, names, cols, bound=100, signed=False, harm=False, cost=False,
         decimals=0, labels=None, colorbar=True, barlabel=None, annotations=True):
    a = np.array(values, dtype=float)
    assert np.isfinite(a).all()
    assert abs(a).max() <= bound + 1e-6, "Colour scale would clip data"
    cm = COST if cost else CHANGE if signed else HARM if harm else SUCCESS
    im = ax.imshow(a, aspect="auto", cmap=cm, vmin=-bound if signed else 0, vmax=bound,
                   interpolation="nearest")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(names)), names)
    ax.tick_params(length=0, pad=5)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", lw=.75)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if annotations:
        for i in range(len(names)):
            for j in range(len(cols)):
                value = a[i, j]
                label = labels[i][j] if labels is not None else (f"{value:+.{decimals}f}" if signed else f"{value:.{decimals}f}")
                if label in ("+0", "-0"):
                    label = "0"
                ax.text(j, i, label, ha="center", va="center", fontsize=6.8,
                        color=contrast_text(im.cmap(im.norm(value))))
    if colorbar:
        cax = ax.inset_axes([1.025, 0, .022, 1])
        cb = ax.figure.colorbar(im, cax=cax, ticks=[-bound, 0, bound] if signed else [0, bound/2, bound])
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=5.8, length=1.5, pad=1.5)
        if barlabel:
            cb.set_label(barlabel, fontsize=6.0, labelpad=2)
    return im


def policy_points(ax, values, names, policies, xmax=100, xlabel="Safe success (%)", annotate=False):
    values = np.array(values)
    rows(ax, names, xmax, xlabel=xlabel, ticks=np.linspace(0, xmax, 5))
    ax.set_xlim(-.018*xmax, 1.018*xmax)
    if xmax == 50:
        ax.set_xticks([0, 10, 20, 30, 40, 50])
    offsets = np.linspace(-.25, .25, len(policies))
    for i in range(len(names)):
        for j, p in enumerate(policies):
            label, col, mark = POLICIES[p]
            ax.plot(values[i, j], i+offsets[j], marker=mark, ms=3.8,
                    mfc="white" if p=="vanilla" else col, mec=col, mew=.75,
                    linestyle="none", zorder=3)
            if annotate:
                ax.text(values[i,j]+xmax*.025, i+offsets[j], f"{values[i,j]:.1f}", va="center", fontsize=6.2)


def paired_counts(ax, vals, names, xmax, xlabel):
    rows(ax, names, xmax, xlabel=xlabel)
    for i, (v, g) in enumerate(vals):
        ax.plot([v, g], [i, i], color=RISK if g>v else GREY, lw=1.05, zorder=2)
        ax.plot(v, i, "o", ms=4, mfc="white", mec=GREY, mew=.9, zorder=3)
        ax.plot(g, i, "D", ms=3.5, color=BLUE, zorder=4)


def box(ax, x, y, w, h, text, face=FAINT, edge=GRID, size=7.1, weight="normal"):
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0.007,rounding_size=.015",
                             facecolor=face, edgecolor=edge, lw=.6, clip_on=False))
    ax.text(x+w/2, y+h/2, text, fontsize=size, weight=weight, ha="center", va="center", linespacing=1.4)


def arrow(ax, start, end, color=MUTED, style="-", **kw):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=.8, linestyle=style,
                                mutation_scale=7, shrinkA=1, shrinkB=1, **kw))


class Designer:
    def __init__(self, data, diagnostics, resources, strata, out):
        configure()
        self.d, self.g = data, diagnostics
        self.resources, self.strata, self.out = resources, strata, Path(out)
        self.primary = tuple(data["primary"]["models"])
        self.figures, self.audit = [], []

    def report(self, campaign, arm, model):
        return self.d[campaign]["reports"][arm+"/"+model]

    def finish(self, fig, name):
        # The NeurIPS text block reduces a 183-mm asset to approximately 140 mm.
        # A 6.6-pt source floor keeps ordinary labels at roughly 5 pt after that reduction.
        for t in fig.findobj(matplotlib.text.Text):
            if t.get_fontsize() < 6.6:
                t.set_fontsize(6.6)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        outside = []
        fonts = set()
        for t in fig.findobj(matplotlib.text.Text):
            if not t.get_visible() or not t.get_text().strip():
                continue
            fonts.add(t.get_fontproperties().get_name())
            bb = t.get_window_extent(renderer)
            # Axis auto-locator may instantiate non-drawn labels outside limits.
            if t.axes is None and not t in fig.texts:
                continue
            if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.x1+1 or bb.y1 > fig.bbox.y1+1:
                outside.append(t.get_text())
        if outside:
            raise AssertionError(name+" text outside canvas: "+repr(outside))
        if not fonts or not fonts.issubset(TIMES_FAMILIES):
            raise AssertionError(name+" unexpected font families: "+repr(fonts))
        for ext in ("pdf", "svg", "png"):
            fig.savefig(self.out/"figures"/(name+"."+ext), dpi=300)
        self.figures.append((name, fig))
        self.audit.append(dict(figure=name, width_mm=round(fig.get_figwidth()*25.4,2),
                               height_mm=round(fig.get_figheight()*25.4,2),
                               minimum_text_pt=6.6, outside_text=outside,
                               font_families=sorted(fonts)))
        print("Designed", name, flush=True)

    def overview(self):
        fig=canvas(164)
        ax=panel(fig,[.055,.725,.89,.205],"a","MedToolFault: evaluate answers and clinical state")
        off(ax)
        box(ax,.00,.44,.17,.35,"Patient-specific\nclinical task")
        box(ax,.225,.44,.15,.35,"LLM agent")
        box(ax,.43,.33,.285,.57,"MedGuard\nVerify reads · Attribute writes\nBound recovery",face=PALE,edge=BLUE,size=7.0)
        box(ax,.77,.44,.225,.35,"FHIR EHR sandbox\nAuthoritative state")
        for s,e in [((.17,.61),(.225,.61)),((.375,.61),(.43,.61)),((.715,.61),(.77,.61))]: arrow(ax,s,e)
        ax.text(.882,.96,"Controlled tool faults",ha="center",fontsize=6.6)
        arrow(ax,(.882,.91),(.882,.80),RISK)
        ax.plot([.01,.99],[.18,.18],color=GRID,lw=.6)
        ax.text(.01,.03,"Independent evaluation",fontsize=6.9,weight="bold",va="center")
        ax.text(.29,.03,"Answer + required state change + no forbidden mutation",fontsize=7.0,va="center")

        ax=panel(fig,[.055,.373,.515,.264],"b","MedToolFault: six tool-failure classes",
                 "Fault manifestation                                      Guard mechanism")
        off(ax)
        tax=[("Interface drift","Parameter rejected","Metadata-guided repair"),
             ("Transient failure","Temporary service error","Bounded retry"),
             ("Stale / incomplete","Old result or missing pages","Re-read and paginate"),
             ("Patient mismatch","Wrong patient reference","Identity check and re-read"),
             ("Provenance conflict","Unverified or conflicting result","Check finalized evidence"),
             ("Ambiguous write","Unknown commit outcome","Compare pre/post records")]
        for i,(name,symptom,check) in enumerate(tax):
            y=.935-i*.163
            if i%2==0: ax.add_patch(Rectangle((-.005,y-.081),1.01,.158,fc=FAINT,ec="none"))
            ax.text(.018,y+.022,name,fontsize=6.8,weight="bold",va="center")
            ax.text(.018,y-.030,symptom,fontsize=6.0,color=MUTED,va="center")
            arrow(ax,(.50,y),(.565,y))
            ax.text(.598,y,check,fontsize=6.5,va="center")

        ax=panel(fig,[.66,.373,.285,.264],"c","MedGuard: attribute write outcomes")
        off(ax)
        box(ax,.08,.80,.84,.16,"Snapshot → write → timeout",face="#FBF0E9",edge=RISK,size=7)
        arrow(ax,(.5,.79),(.5,.68))
        ax.text(.5,.62,"Read authoritative state",ha="center",fontsize=6.9)
        arrow(ax,(.5,.56),(.22,.44)); arrow(ax,(.5,.56),(.78,.44))
        box(ax,.01,.19,.42,.25,"One new match\nDo not repeat",face=PALE,edge=BLUE,size=6.6)
        box(ax,.57,.19,.42,.25,"No new match\nBounded retry",face=FAINT,size=6.6)
        ax.text(.5,.020,"Requires visibility and attribution",ha="center",fontsize=6.2)

        ax=panel(fig,[.055,.085,.89,.18],"d","Five complementary evaluations",
                 "127,861 formal episodes; repeated evaluations, not independent patients")
        off(ax)
        names=[("PRIMARY","9 configurations","primary"),("MECHANISMS","Controls and stress","a"),
               ("TRANSFER","Expanded tasks","c"),("RAW-CALL CAP","12 calls per episode","d"),("EXTERNAL QA","Original answers","b")]
        counts=list(self.d["counts"].values())
        for i,((label,detail,_),count) in enumerate(zip(names,counts)):
            x=i*.202
            ax.add_patch(Rectangle((x,.13),.184,.79,facecolor=PALE if i==0 else FAINT,edgecolor="none"))
            ax.plot([x,x+.184],[.92,.92],color=BLUE if i==0 else GREY,lw=1.3)
            ax.text(x+.092,.72,label,ha="center",fontsize=6.4,weight="bold")
            ax.text(x+.092,.43,f"{count:,}",ha="center",fontsize=10)
            ax.text(x+.092,.23,detail,ha="center",fontsize=6.2)
        fig.text(.5,.035,"Public-interface verification only · No hidden answers · No clinical deployment claim",
                 ha="center",fontsize=6.5,color=MUTED)
        self.finish(fig,"fig01_overview")

    def primary_results(self):
        fig=canvas(169); names=[SHORT[m] for m in self.primary]
        ax=panel(fig,[.17,.696,.315,.23],"a","Clean competence and recovery","Raw safe success; clean n = 255, fault n = 806")
        rows(ax,names,100,xlabel="Safe success (%)",ticks=[0,25,50,75,100])
        for i,m in enumerate(self.primary):
            c=self.d["primary"]["models"][m]["conditions"]
            clean=c["vanilla"]["clean_success"]["rate"]*100
            v=c["vanilla"]["fault_success"]["rate"]*100; g=c["medguard"]["fault_success"]["rate"]*100
            ax.plot([v,g],[i,i],color=OTHER,lw=1.5)
            ax.plot(clean,i-.20,marker="|",color=INK,ms=5,ls="none")
            ax.plot(v,i,marker="o",mfc="white",mec=GREY,ms=3.5,ls="none")
            ax.plot(g,i,marker="D",color=BLUE,ms=3.5,ls="none")
        ax=panel(fig,[.622,.696,.335,.23],"b","Paired recovery, two estimands","MedGuard − vanilla; 95% patient-cluster intervals")
        forest(ax,[""]*len(names),(-6,61),[0,20,40,60])
        for i,m in enumerate(self.primary):
            r=self.d["primary"]["models"][m]
            dot(ax,i-.14,r["paired_guard_gain"],GREY,"o",hollow=True)
            dot(ax,i+.14,r["paired_guard_gain_clean_capable"],BLUE)
        legend(fig,[key("Clean vanilla",INK,"|"),policy_key("vanilla"),policy_key("medguard")],.326,.624)
        legend(fig,[key("All tasks",GREY,"o",True),key("Clean-capable tasks",BLUE,"D")],.79,.624)
        ax=panel(fig,[.17,.345,.735,.205],"c","Where does recovery occur?","Raw MedGuard − vanilla difference (percentage points)")
        heat(ax,[[100*(self.d["primary"]["models"][m]["by_hazard"]["medguard"][h]["rate"]-
                        self.d["primary"]["models"][m]["by_hazard"]["vanilla"][h]["rate"]) for h in HAZARDS] for m in self.primary],
             names,HAZARD_LABELS,signed=True,bound=100,decimals=1)
        ax=panel(fig,[.17,.080,.315,.17],"d","Unsafe state is a distinct endpoint","Action-required faults; n = 202, lower is better")
        paired_counts(ax,[[self.d["primary"]["models"][m]["conditions"][p]["persistent_unsafe_action_required"]["count"] for p in ("vanilla","medguard")] for m in self.primary],names,202,"Unsafe-state episodes")
        ax.set_xticks([0,50,100,150,200]); ax.tick_params(axis="y",labelsize=6.6)
        ax=panel(fig,[.622,.080,.335,.17],"e","Ambiguous committed writes","Committed timeouts; n = 60, lower is better")
        policy_points(ax,[[self.d["primary"]["models"][m]["conditions"][p]["commit_timeout_duplicates"]["count"] for p in ("vanilla","retry_only","medguard")] for m in self.primary],
                      [""]*len(names),("vanilla","retry_only","medguard"),60,"Duplicate-creation episodes")
        legend(fig,[policy_key("vanilla"),policy_key("medguard"),key("Increase in unsafe state",RISK,"_")],.325,.018)
        legend(fig,[policy_key(p) for p in ("vanilla","retry_only","medguard")],.79,.018)
        self.finish(fig,"fig02_primary")

    def mechanisms(self):
        fig=canvas(135); names=[SHORT[m] for m in MODELS]
        ax=panel(fig,[.165,.626,.30,.28],"a","MedGuard and matched controls","Matched-control safe success; n = 806 per policy")
        policies=("vanilla","informed","evidence_only","medguard")
        policy_points(ax,[[self.report("a","matched_controls",m)["conditions"][p]["fault_success"]["rate"]*100 for p in policies] for m in MODELS],names,policies)
        ax=panel(fig,[.645,.626,.30,.28],"b","MedGuard versus informed control","Paired gain and 95% patient-cluster interval")
        forest(ax,names,(-8,26),[-5,0,10,20])
        for i,m in enumerate(MODELS): dot(ax,i,self.report("a","matched_controls",m)["comparisons"]["informed-medguard"],reverse=True)
        legend(fig,[policy_key(p) for p in policies],.335,.52,ncol=2)
        ax=panel(fig,[.165,.148,.30,.25],"c","Which guard components matter?","Deletion − full MedGuard; negative = lower success")
        forest(ax,names,(-24,10),[-20,-10,0,10])
        comps=[("no_read",BLUE,"o"),("no_write",RISK,"s"),("no_retry",AMBER,"^")]
        for i,m in enumerate(MODELS):
            for j,(p,c,mk) in enumerate(comps): dot(ax,i+(j-1)*.23,self.report("a","matched_controls",m)["comparisons"][p+"-medguard"],c,mk)
        ax=panel(fig,[.645,.148,.30,.25],"d","Does recovery survive a hard budget?","12 raw calls, including helpers; n = 160 faults")
        policy_points(ax,[[self.report("d","strict_raw_budget_12",m)["conditions"][p]["fault_success"]["rate"]*100 for p in policies] for m in MODELS],names,policies)
        legend(fig,[key("Read verification",BLUE),key("Write attribution",RISK,"s"),key("Generic retry",AMBER,"^")],.31,.049,ncol=2)
        legend(fig,[policy_key(p) for p in policies],.785,.049,ncol=2)
        self.finish(fig,"fig03_mechanisms")

    def stress(self):
        fig=canvas(145); names=[SHORT[m] for m in MODELS]
        stress=["operation_matched","repeated_faults","compound_faults","concurrent_writer","visibility_delay","calls_24","tokens_1024","tokens_4096"]
        labels=["Operation-matched","Repeated faults","Compound faults","Concurrent writer","Delayed visibility","24-call ceiling","1,024 output tokens","4,096 output tokens"]
        ax=panel(fig,[.21,.565,.30,.34],"a","Stress and budget sensitivity","Paired MedGuard − vanilla gain (pp)")
        heat(ax,[[self.report("a",s,m)["comparisons"]["medguard-vanilla"]["difference"]*100 for m in MODELS] for s in stress],labels,names,bound=60,signed=True,decimals=1)
        ax.tick_params(axis="x",labelsize=6.1)
        ax=panel(fig,[.69,.565,.265,.34],"b","Sensitivity to sampled decoding","Three seeds; fixed 50-task subset")
        forest(ax,names,(-20,50),[-20,0,20,40])
        seeds=[("sampling_20260918",GREY,"o"),("sampling_20260919",AMBER,"^"),("sampling_20260920",BLUE,"s")]
        for i,m in enumerate(MODELS):
            for j,(s,c,mk) in enumerate(seeds): dot(ax,i+(j-1)*.24,self.report("a",s,m)["comparisons"]["medguard-vanilla"],c,mk)
        legend(fig,[key("Seed "+str(i+1),c,mk) for i,(_,c,mk) in enumerate(seeds)],.817,.458)
        ax=panel(fig,[.21,.147,.30,.23],"c","The consistency boundary","Success count / assigned episodes")
        vals=[]; lab=[]
        for m in MODELS:
            cells=[self.report("a",s,m)["conditions"][p]["fault_success"] for s in ("visibility_delay","concurrent_writer") for p in ("vanilla","medguard")]
            vals.append([100*c["rate"] for c in cells]); lab.append([f'{c["count"]}/{c["n"]}' for c in cells])
        heat(ax,vals,names,["Delay\nV","Delay\nMG","Writer\nV","Writer\nMG"],labels=lab)
        ax=panel(fig,[.69,.147,.265,.23],"d","Fault placement changes the estimate","Each arm has its own paired comparison")
        forest(ax,names,(-10,30),[-10,0,10,20,30])
        for i,m in enumerate(MODELS):
            dot(ax,i-.13,self.report("a","matched_controls",m)["comparisons"]["medguard-vanilla"],GREY,"o",hollow=True)
            dot(ax,i+.13,self.report("a","operation_matched",m)["comparisons"]["medguard-vanilla"],BLUE)
        legend(fig,[key("Standard",GREY,"o",True),key("Operation-matched",BLUE,"D")],.814,.047,ncol=2)
        fig.text(.21,.047,"V, vanilla; MG, MedGuard. All guarded delay cases fail.",fontsize=6.3,color=MUTED)
        self.finish(fig,"fig04_stress")

    def transfer(self):
        fig=canvas(165); names=[SHORT[m] for m in MODELS]
        arms=["decision_branches","preexisting_equivalent","preexisting_different","preexisting_cancelled","external_actions"]
        ax=panel(fig,[.19,.705,.305,.216],"a","MedToolFault task expansion","Paired MedGuard − vanilla gain (pp)")
        heat(ax,[[self.report("c",s,m)["comparisons"]["medguard-vanilla"]["difference"]*100 for m in MODELS] for s in arms],
             ["Decision branches","Equivalent record","Different record","Cancelled record","External actions"],names,bound=60,signed=True,decimals=1)
        ax.tick_params(axis="x",labelsize=6.1)
        ax=panel(fig,[.665,.705,.275,.216],"b","Required-action state integrity","Unsafe state / action-required faults")
        vals=[]; labels=[]
        for s in ("decision_branches","external_actions"):
            for p in ("vanilla","medguard"):
                c=[self.report("c",s,m)["conditions"][p]["action_required_unsafe_state"] for m in MODELS]
                vals.append([100*x["rate"] for x in c]); labels.append([f'{x["count"]}\n/{x["n"]}' for x in c])
        heat(ax,vals,["Branches / V","Branches / MG","External / V","External / MG"],names,harm=True,labels=labels)
        ax.tick_params(axis="x",labelsize=6.1)
        ax=panel(fig,[.19,.405,.305,.177],"c","Secondary paired-outcome analysis","Qwen external actions; paired transitions")
        transitions=["success_to_success","failure_to_success","success_to_failure","failure_to_failure"]
        colours=[BLUE,"#AFC9DE",RISK,OTHER]
        for i,st in enumerate(("clean","fault")):
            r=self.g["reports"]["c_qwen_external_actions"]["paired"][st]; left=0
            for t,col in zip(transitions,colours):
                n=r["transitions"].get(t,0); w=100*n/r["n"]
                ax.barh(i,w,left=left,height=.42,color=col,edgecolor="white",lw=.5)
                if w>6: ax.text(left+w/2,i,str(n),ha="center",va="center",fontsize=7,color=contrast_text(to_rgb(col)))
                left+=w
        rows(ax,["Clean (138)","Fault (732)"],100,xlabel="Paired assignments (%)",ticks=[0,25,50,75,100],bands=False)
        ax.grid(False)
        ax=panel(fig,[.665,.405,.275,.177],"d","Post-hoc verification-error markers","Overlapping tags among newly lost pairs")
        off(ax)
        labels=["Unsupported query parameter","Generic verification error","Repeated identical GET","Claimed done; write missing"]
        tags=["raw_unknown_system_parameter","generic_verification_error","same_agent_get_at_least_three_times","claimed_completed_with_missing_write"]
        ax.text(.99,1.02,"Clean / fault",ha="right",fontsize=6.1,color=MUTED)
        for i,(lab,tag) in enumerate(zip(labels,tags)):
            y=.87-i*.238
            ax.plot([.0,.99],[y-.105,y-.105],color=GRID,lw=.55)
            ax.text(.01,y,lab,fontsize=6.2,va="center")
            v=[self.g["reports"]["c_qwen_external_actions"]["paired"][st]["medguard_loss_tags"][tag] for st in ("clean","fault")]
            ax.text(.99,y,f"{v[0]} / {v[1]}",ha="right",fontsize=6.5,va="center",weight="bold")
        legend(fig,[Patch(fc=c,label=l) for c,l in zip(colours,["Retained","Rescued","Lost","Both fail"])],.34,.315,ncol=4)
        ax=panel(fig,[.055,.075,.89,.165],"e","Verification-error pathway in the selected transfer contrast","Post-hoc trace pattern; not a tested repair or causal intervention")
        off(ax)
        items=[("1","Invalid request","Unsupported\nquery parameter",FAINT),
               ("2","Information removed","Specific error replaced\nby verification failure",PALE),
               ("3","No useful correction","Identical invalid\nquery repeated",FAINT),
               ("4","Task remains incomplete","Completion claimed;\nrequired write absent","#FBF0E9")]
        for i,(number,heading,body,face) in enumerate(items):
            x=i*.255
            ax.add_patch(Rectangle((x,.08),.225,.85,fc=face,ec="none"))
            ax.text(x+.016,.77,number,fontsize=9,weight="bold")
            ax.text(x+.016,.55,heading,fontsize=6.6,weight="bold")
            ax.text(x+.016,.25,body,fontsize=6.5,linespacing=1.4,va="center")
            if i<3: arrow(ax,(x+.226,.48),(x+.254,.48))
        fig.text(.5,.025,"All 56 clean and 234 fault losses contain these tags; co-occurrence does not establish repair efficacy.",ha="center",fontsize=6.4,color=MUTED)
        self.finish(fig,"fig05_transfer")

    def external(self):
        fig=canvas(145); names=[SHORT[m] for m in MODELS]
        ax=panel(fig,[.19,.635,.29,.27],"a","External original-answer accuracy","100 questions × five faults; 45 patients")
        policies=("vanilla","informed","medguard")
        policy_points(ax,[[self.d["b"][m]["conditions"][p]["correct_fault"]/5 for p in policies] for m in MODELS],names,policies,50,"Fault-answer accuracy (%)")
        ax=panel(fig,[.665,.635,.29,.27],"b","MedGuard versus baseline policies","Patient-equal paired effect and 95% interval")
        forest(ax,names,(-9,19),[-5,0,5,10,15])
        for i,m in enumerate(MODELS):
            dot(ax,i-.14,self.d["b"][m]["comparisons"]["medguard-vanilla"],BLUE,"D")
            dot(ax,i+.14,self.d["b"][m]["comparisons"]["medguard-informed"],AMBER,"^")
        legend(fig,[policy_key(p) for p in policies],.335,.537)
        legend(fig,[key("MG − vanilla",BLUE,"D"),key("MG − informed",AMBER,"^")],.81,.537)
        ax=panel(fig,[.19,.165,.29,.23],"c","Read-side failure heterogeneity","Raw MedGuard − vanilla gain (pp)")
        hazards=["interface_drift","transient_503","partial_history","patient_mismatch","provenance_conflict"]
        heat(ax,[[self.d["b"][m]["conditions"]["medguard"]["by_hazard"][h]["correct"]-
                   self.d["b"][m]["conditions"]["vanilla"]["by_hazard"][h]["correct"] for m in MODELS] for h in hazards],
             ["Interface drift","Transient 503","Partial history","Patient mismatch","Provenance conflict"],names,bound=25,signed=True,decimals=0)
        ax.tick_params(axis="x",labelsize=6.1)
        ax=panel(fig,[.665,.165,.29,.23],"d","Termination is not correctness","600 clean + fault episodes per policy")
        ypos=[]; labs=[]
        for i,m in enumerate(MODELS):
            for j,p in enumerate(policies):
                y=i*4+j; left=0; ypos.append(y); labs.append((SHORT[m]+"  " if j==1 else "")+{"vanilla":"V","informed":"I","medguard":"MG"}[p])
                for s,col in (("final",OTHER),("turn_limit",AMBER),("context_limit",RISK)):
                    w=self.d["b"][m]["conditions"][p]["stop_reasons"].get(s,0)/6
                    ax.barh(y,w,left=left,color=col,height=.75,edgecolor="white",lw=.35); left+=w
                assert abs(left-100)<1e-6
        ax.set(yticks=ypos,yticklabels=labs,xlim=(0,100),ylim=(14.7,-.8),xlabel="Episodes (%)")
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y",labelsize=6.4)
        legend(fig,[Patch(fc=c,label=l) for l,c in (("Final response",OTHER),("Turn limit",AMBER),("Context limit",RISK))],.81,.058,ncol=3)
        fig.text(.19,.058,"MG–informed intervals all cross zero; not proof of equivalence.",fontsize=6.3,color=MUTED)
        self.finish(fig,"fig06_external")

    def resource_fig(self):
        fig=canvas(139); names=[SHORT[m] for m in MODELS]
        stages=("A","D","B-lite")
        def val(stage, model, policy, field):
            return next(r[field] for r in self.resources if r["stage"]==stage and r["model"]==model and r["policy"]==policy)
        for rect,field,letter,heading in [([.17,.66,.29,.245],"tokens","a","Token savings depend on the workload"),
                                         ([.65,.66,.29,.245],"raw_calls","b","Verification adds environment work")]:
            ax=panel(fig,rect,letter,heading,"Change in mean consumption: MedGuard versus vanilla (%)")
            vals=[[100*(val(s,m,"medguard",field)/val(s,m,"vanilla",field)-1) for s in stages] for m in MODELS]
            heat(ax,vals,names,["Controls\n(A)","12-call cap\n(D)","External QA\n(B-lite)"],bound=80,signed=True,cost=True,decimals=1)
        for rect,field,letter,heading in [([.17,.16,.29,.32],"tokens","c","Absolute accumulated token counts"),
                                         ([.65,.16,.29,.32],"raw_calls","d","Absolute raw-call counts")]:
            ax=panel(fig,rect,letter,heading,"Connected means; lines are not confidence intervals")
            labs=[]; ys=[]
            for i,m in enumerate(MODELS):
                for j,s in enumerate(stages):
                    y=i*4+j; ys.append(y); labs.append((SHORT[m]+"  " if j==1 else "")+s)
                    scale=1000 if field=="tokens" else 1
                    v=val(s,m,"vanilla",field)/scale; g=val(s,m,"medguard",field)/scale
                    ax.plot([v,g],[y,y],color=GREY,lw=.9)
                    ax.plot(v,y,"o",mfc="white",mec=GREY,ms=3.5,mew=.8)
                    ax.plot(g,y,"D",color=BLUE,ms=3.4)
                if i%2==0: ax.axhspan(i*4-.5,i*4+2.5,color=FAINT,zorder=-2,lw=0)
            ax.set(yticks=ys,yticklabels=labs,ylim=(14.7,-.8),xlim=(0,80 if field=="tokens" else 18))
            ax.set_xlabel("Accumulated tokens (thousands)" if field=="tokens" else "Raw calls per fault episode")
            ax.spines["left"].set_visible(False); ax.grid(axis="x",color=GRID,lw=.45)
            ax.tick_params(axis="y",labelsize=6.6)
        legend(fig,[policy_key("vanilla"),policy_key("medguard")],.51,.05)
        fig.text(.5,.015,"Tokens include repeated input history. No inference about latency, energy use or financial cost.",ha="center",fontsize=6.3,color=MUTED)
        self.finish(fig,"fig07_resources")

    def supplementary(self):
        fig=canvas(140); names=[SHORT[m] for m in self.primary]
        for rect,p,l in [([.17,.60,.735,.31],"vanilla","a"),([.17,.13,.735,.31],"medguard","b")]:
            ax=panel(fig,rect,l,("Vanilla" if p=="vanilla" else "MedGuard")+": heterogeneity across task templates","Raw safe success under assigned faults (%)")
            heat(ax,[[100*self.d["primary"]["models"][m]["conditions"][p]["by_template"][str(t)]["fault_success"]["rate"] for t in range(1,11)] for m in self.primary],names,[str(t) for t in range(1,11)],decimals=1)
            ax.set_xlabel("Source task template",labelpad=5)
        self.finish(fig,"figS01_templates")
        fig=canvas(138)
        tags=["no_successful_retrieval","no_successful_code_call","code_tool_error","any_generation_limit","stopped_context_limit","stopped_turn_limit"]
        names=["No successful retrieval","No successful code call","Any code-tool error","Generation limit","Context stop","Turn stop"]
        rects=([.23,.61,.23,.30],[.715,.61,.23,.30],[.23,.15,.23,.30],[.715,.15,.23,.30])
        for rect,m,l in zip(rects,MODELS,"abcd"):
            ax=panel(fig,rect,l,SHORT[m]+": observed workflow markers","Fault episodes with each tag (%); tags overlap")
            heat(ax,[[100*self.g["reports"]["b_"+m]["cells"][p+"/fault"]["tags"].get(t,0)/500 for p in ("vanilla","informed","medguard")] for t in tags],names,["Vanilla","Informed","MedGuard"],harm=True,decimals=1)
            ax.tick_params(axis="x",labelsize=6.1)
        fig.text(.5,.035,"Observed tags are not mutually exclusive causes. A task need not require a code-tool call.",ha="center",fontsize=6.5,color=MUTED)
        self.finish(fig,"figS02_workflow_tags")
        fig=canvas(94)
        for rect,sel,heading,policies,letter in [([.21,.27,.25,.53],"c_ministral_decision_branches","Ministral: decision branches",("vanilla","medguard"),"a"),
                                                ([.70,.27,.25,.53],"a_llama_matched","Llama: matched controls",("vanilla","informed","medguard"),"b")]:
            ax=panel(fig,rect,letter,heading,"Required action versus no action")
            labs=[]; ys=[]
            for i,st in enumerate(("action","no_action")):
                for j,p in enumerate(policies):
                    row=next(r for r in self.strata if r["selection"]==sel and r["stratum"]==st and r["policy"]==p)
                    y=i*(len(policies)+1)+j; ys.append(y)
                    labs.append(("Action" if st=="action" else "No action")+" / "+{"vanilla":"V","informed":"I","medguard":"MG"}[p]+f' (n={row["n"]})')
                    left=0
                    for field,col in (("success",BLUE),("unsafe",RISK),("other_failure",OTHER)):
                        n=row[field]; width=100*n/row["n"]
                        ax.barh(y,width,left=left,color=col,height=.70,edgecolor="white",lw=.4)
                        if width>9: ax.text(left+width/2,y,str(n),ha="center",va="center",fontsize=7,color=contrast_text(to_rgb(col)))
                        left+=width
            ax.set(yticks=ys,yticklabels=labs,ylim=(max(ys)+.7,-.7),xlim=(0,100),xlabel="Assigned fault episodes (%)")
            ax.spines["left"].set_visible(False); ax.tick_params(axis="y",labelsize=6.7)
        legend(fig,[Patch(fc=c,label=l) for l,c in (("Safe success",BLUE),("Unsafe state",RISK),("Other failure",OTHER))],.5,.115)
        fig.text(.5,.035,"Within each row, outcomes are mutually exclusive and sum to the assigned denominator.",ha="center",fontsize=6.5,color=MUTED)
        self.finish(fig,"figS03_action_strata")

    def run(self):
        self.overview(); self.primary_results(); self.mechanisms(); self.stress()
        self.transfer(); self.external(); self.resource_fig(); self.supplementary()
        with PdfPages(self.out/"figures/ALL_ANALYSIS_FIGURES.pdf") as pdf:
            for _, fig in self.figures:
                pdf.savefig(fig)
                plt.close(fig)
        (self.out/"figure_design_audit.json").write_text(json.dumps({
            "version": "2026-10-06-times-typography", "figures": self.audit,
            "fixed_colour_semantics": {"method": BLUE,"unsafe_or_adverse": RISK,"vanilla": GREY,"informed": AMBER},
            "colour_scales_checked_for_clipping": True,
            "source_data_modified": False,
            "note": "Layout checks and design references are not journal endorsement or acceptance criteria."
        },indent=2)+"\n")
        return len(self.figures)


def render_all(data, diagnostics, resources, strata, out):
    return Designer(data, diagnostics, resources, strata, out).run()
