"""Render the requested local progress report and a data-derived SVG, offline."""
import argparse
import html
import json
from pathlib import Path

from markdown_it import MarkdownIt

LABELS = {"qwen35_9b": "Qwen3.5-9B", "ministral3_8b": "Ministral-3-8B", "qwen25_7b": "Qwen2.5-7B",
          "qwen25_14b": "Qwen2.5-14B", "qwen3_8b": "Qwen3-8B", "qwen3_8b_thinking": "Qwen3-8B (thinking)",
          "llama31_8b": "Llama-3.1-8B", "gemma3_12b": "Gemma-3-12B", "phi4_mini": "Phi-4-mini"}

TEXT = {
    "zh": {
        "locale": "zh-CN", "page_title": "MedToolBench 项目进展报告 · 2026-09-15",
        "filename": "interim_fault_success.svg", "alt": "中期故障成功率比较",
        "title": "已审计共同测试子集：故障下的安全成功率（中期）",
        "desc": "每个模型使用同一批已完成任务，比较无防护与 MedGuard；不是最终全量结果。",
        "heading": "故障下的安全成功率：无防护 vs. MedGuard",
        "cohort": "中期描述性结果 · {tasks} 个任务 · {groups} 个患者/查询组 · 每模型 {faults} 个故障变体",
        "vanilla": "无防护",
        "foot1": "纳入未触发的故障分配；未将故障变体视为独立患者。此图为原始比例，主要任务加权指标见报告。",
        "foot2": "不代表临床伤害发生率、全量最终结果或外部数据集模型效果。",
    },
    "en": {
        "locale": "en", "page_title": "MedToolBench Progress Report · 2026-09-15",
        "filename": "interim_fault_success_en.svg", "alt": "Interim fault safe-success comparison",
        "title": "Audited common test subset: safe success under faults (interim)",
        "desc": "Vanilla and MedGuard evaluated on the same completed tasks for every model. Not final full-study results.",
        "heading": "Safe success under faults: Vanilla vs. MedGuard",
        "cohort": "Descriptive interim results · {tasks} tasks · {groups} patient/lookup groups · {faults} fault variants per model per condition",
        "vanilla": "Vanilla",
        "foot1": "All assigned faults included, even if untriggered. Raw proportions shown; task-weighted effects are reported separately.",
        "foot2": "Variants are not independent patients. These are not clinical harm rates, final results, or external-model results.",
    },
}


def chart(metrics, lang="zh"):
    t = TEXT[lang]
    selection = metrics["selection"]
    counts = selection["hazard_variants_per_system_condition"]
    denominator = sum(n for h, n in counts.items() if h != "clean")
    cohort = t["cohort"].format(tasks=selection["tasks"], groups=selection["patient_groups"], faults=denominator)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 730" role="img" aria-labelledby="title desc">',
           f'<title id="title">{html.escape(t["title"])}</title>',
           f'<desc id="desc">{html.escape(t["desc"])}</desc>',
           '<rect width="1160" height="730" fill="white"/>',
           '<style>text{font-family:Arial,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;fill:#203047}</style>',
           f'<text x="32" y="40" font-size="23" font-weight="bold">{html.escape(t["heading"])}</text>',
           f'<text x="32" y="70" font-size="16">{html.escape(cohort)}</text>',
           f'<rect x="32" y="89" width="18" height="13" fill="#8896a8"/><text x="58" y="101" font-size="15">{html.escape(t["vanilla"])}</text>',
           '<rect x="170" y="89" width="18" height="13" fill="#217c9d"/><text x="196" y="101" font-size="15">MedGuard</text>']
    for pct in range(0, 101, 20):
        x = 255 + pct * 7
        svg += [f'<line x1="{x}" x2="{x}" y1="125" y2="642" stroke="#e1e7ed"/>',
                f'<text x="{x}" y="665" text-anchor="middle" font-size="14">{pct}%</text>']
    for i, (name, summary) in enumerate(metrics["models"].items()):
        y = 145 + 56 * i
        svg.append(f'<text x="239" y="{y + 14}" text-anchor="end" font-size="16">{html.escape(LABELS[name])}</text>')
        for offset, condition, color in ((0, "vanilla", "#8896a8"), (21, "medguard", "#217c9d")):
            value = summary["conditions"][condition]["fault_success"]
            width = value["rate"] * 700
            svg += [f'<rect x="255" y="{y + offset - 10}" width="{width:.3f}" height="15" fill="{color}" rx="2"/>',
                    f'<text x="{263 + width:.3f}" y="{y + offset + 2}" font-size="13">{100 * value["rate"]:.1f}% ({value["count"]}/{value["n"]})</text>']
    svg += [f'<text x="32" y="696" font-size="14">{html.escape(t["foot1"])}</text>',
            f'<text x="32" y="719" font-size="14">{html.escape(t["foot2"])}</text>', '</svg>']
    return "\n".join(svg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--markdown", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--lang", choices=tuple(TEXT), default="zh")
    p.add_argument("--no-chart", action="store_true", help="Render a text/table report without a chart asset")
    p.add_argument("--compact", action="store_true", help="Compact spacing for a short briefing")
    a = p.parse_args()
    t = TEXT[a.lang]
    if a.out.exists():
        raise SystemExit("Use new report asset paths")
    data = json.loads(a.metrics.read_text())
    drawing = chart(data, a.lang)
    svg_path = a.out.with_name(t["filename"])
    if not a.no_chart and svg_path.exists() and svg_path.read_text() != drawing:
        raise SystemExit("Existing SVG differs; use a new report asset path")
    body = MarkdownIt("commonmark", {"html": False}).enable("table").render(a.markdown.read_text())
    chart_tag = f'<img src="{t["filename"]}" alt="{t["alt"]}" />'
    if a.no_chart and chart_tag in body:
        raise SystemExit("The document requests a chart; remove --no-chart")
    if not a.no_chart:
        body = body.replace(chart_tag, drawing)
    page = f'''<!doctype html><html lang="{t["locale"]}"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(t["page_title"])}</title><style>
''' + '''
body{margin:0;background:#edf2f7;color:#253245;font:16px/1.85 system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}
main{max-width:1050px;margin:30px auto;padding:44px 56px;background:white;border-top:6px solid #217c9d}
h1{font-size:29px;line-height:1.45}h2{font-size:23px;margin-top:42px;border-bottom:1px solid #dce5ed;padding-bottom:8px}h3{font-size:19px;margin-top:27px}
table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}th,td{padding:8px 10px;border:1px solid #dce5ed;text-align:left}th{background:#edf5f8}tr:nth-child(even){background:#f8fafc}
blockquote{margin:20px 0;padding:10px 20px;border-left:4px solid #217c9d;background:#f0f7fa}a{color:#146481}code{font:13px monospace;overflow-wrap:anywhere}pre{white-space:pre-wrap;background:#f4f6f9;padding:16px}
svg,img{max-width:100%;height:auto}strong{color:#132f46}li{margin:5px 0}
@media print{body{background:white;font-size:11pt}main{margin:0;padding:0;border:none;max-width:none}h2,h3{break-after:avoid}tr,blockquote,svg{break-inside:avoid}a{color:inherit;text-decoration:none}@page{size:A4;margin:17mm}}
</style><main>''' + body + '</main></html>'
    if a.compact:
        page = page.replace('</style>', '''
body{line-height:1.55}main{max-width:980px;padding:28px 40px;margin:20px auto}
h3{font-size:18px;margin:18px 0 7px}p{margin:9px 0}table{margin:12px 0;font-size:14px}
th,td{padding:6px 9px}@media print{body{font-size:10.5pt;line-height:1.4}main{padding:0;margin:0}h3{font-size:12pt;margin-top:12px}table{font-size:10pt}@page{margin:14mm}}
</style>''')
    if not a.no_chart and not svg_path.exists():
        svg_path.write_text(drawing)
    a.out.write_text(page)
    print(json.dumps({"html": str(a.out), "svg": None if a.no_chart else str(svg_path)}))


if __name__ == "__main__":
    main()
