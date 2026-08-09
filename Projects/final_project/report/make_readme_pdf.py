"""
Render the project README.md to report/README.pdf.

No pandoc dependency: converts the subset of Markdown the README actually uses
(headings, bold/italic, inline code, fenced code, tables, lists, links, rules,
inline math) into LaTeX and compiles it with pdflatex.

Run: python3 report/make_readme_pdf.py
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "README.md")
OUTDIR = os.path.join(ROOT, "report")
STEM = "README"

PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=0.9in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{fancyvrb}
\usepackage[hidelinks,breaklinks=true]{hyperref}
\usepackage{titlesec}
\usepackage{needspace}

\definecolor{accent}{HTML}{991B1B}
\definecolor{ink}{HTML}{1E293B}
\definecolor{codebg}{HTML}{F6F8FA}

\titleformat{\section}{\Large\bfseries\color{ink}}{}{0pt}{}[\vspace{-0.6em}\textcolor{accent}{\rule{\linewidth}{1pt}}]
\titleformat{\subsection}{\large\bfseries\color{ink}}{}{0pt}{}
\titlespacing*{\section}{0pt}{1.4em}{0.7em}
\titlespacing*{\subsection}{0pt}{1.0em}{0.4em}

\setlist[itemize]{leftmargin=1.3em,itemsep=0.15em,topsep=0.3em}
\setlist[enumerate]{leftmargin=1.5em,itemsep=0.15em,topsep=0.3em}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5em}
\renewcommand{\arraystretch}{1.25}

% \footnotesize keeps the widest code line (85 chars, in the tree) inside the margin
\DefineVerbatimEnvironment{code}{Verbatim}{fontsize=\footnotesize,xleftmargin=1em,frame=leftline,framerule=1.2pt,rulecolor=\color{accent!35},formatcom=\color{ink}}

% long unbreakable \texttt runs (paths, filenames) otherwise punch past the margin
\sloppy
\emergencystretch=3em

\begin{document}
"""

SPECIALS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def esc(t):
    return "".join(SPECIALS.get(c, c) for c in t)


def inline(text):
    """Convert inline Markdown to LaTeX, protecting code and math spans."""
    slots = []

    def stash(rep):
        slots.append(rep)
        return f"\x00{len(slots) - 1}\x00"

    # protect inline math first, then code, so $ inside code is untouched
    text = re.sub(r"\$([^$]+)\$", lambda m: stash(f"${m.group(1)}$"), text)
    def brk(s):
        # \texttt is unbreakable, so long paths punch out of narrow table cells.
        # Offer zero-width break points after path separators and underscores.
        return s.replace("/", "/\\allowbreak{}").replace(r"\_", r"\_\allowbreak{}")

    text = re.sub(r"`([^`]+)`", lambda m: stash(r"\texttt{" + brk(esc(m.group(1))) + "}"), text)

    # links before escaping, so the URL survives intact
    def link(m):
        # escape the label as-is; any \x00N\x00 markers inside it survive esc()
        # untouched and are expanded by the restore loop below
        label, url = m.group(1), m.group(2)
        return stash(r"\href{" + url.replace("%", r"\%").replace("#", r"\#") + "}{" + esc(label) + "}")

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link, text)

    # bare URLs on their own: typeset with \url so they stay clickable and can break
    text = re.sub(r"https?://[^\s)\]]+",
                  lambda m: stash(r"\url{" + m.group(0).replace("%", r"\%") + "}"), text)

    text = esc(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\emph{\1}", text)
    text = text.replace("→", r"$\rightarrow$").replace("↔", r"$\leftrightarrow$")
    text = text.replace("×", r"$\times$").replace("±", r"$\pm$")
    # placeholders can nest (a code span inside a link label), so expand to fixpoint
    while "\x00" in text:
        text = re.sub(r"\x00(\d+)\x00", lambda m: slots[int(m.group(1))], text)
    return text


def table(rows):
    """rows: list of list of cell strings; row 1 is the header, row 2 the separator."""
    header, body = rows[0], rows[2:]
    n = len(header)
    if n == 2:
        widths = [0.34, 0.66]
    else:
        first = 0.31 if n <= 4 else 0.24
        widths = [first] + [(1.0 - first) / (n - 1)] * (n - 1)
    # Fractions sum to 1, so each column must give back its own 2\tabcolsep or the
    # table runs past \linewidth by 2(n-1)\tabcolsep. raggedright as well, since
    # justifying a narrow cell stretches interword space badly around monospace.
    spec = "".join(
        r">{\raggedright\arraybackslash}p{\dimexpr " + f"{w:.4f}" + r"\linewidth-2\tabcolsep\relax}"
        for w in widths)
    out = [r"\begin{center}\small", r"\begin{longtable}{@{}" + spec + r"@{}}", r"\toprule"]
    out.append(" & ".join(r"\textbf{" + inline(c) + "}" for c in header) + r" \\")
    out += [r"\midrule", r"\endhead"]
    for r in body:
        cells = (r + [""] * n)[:n]
        out.append(" & ".join(inline(c) for c in cells) + r" \\")
    out += [r"\bottomrule", r"\end{longtable}", r"\end{center}"]
    return out


def convert(md):
    lines = md.split("\n")
    out, i = [], 0
    list_mode = None

    def close_list():
        nonlocal list_mode
        if list_mode:
            out.append(r"\end{" + list_mode + "}")
            list_mode = None

    while i < len(lines):
        ln = lines[i]

        if ln.startswith("```"):
            close_list()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            # pdflatex with T1 has no glyphs for the box-drawing characters, and
            # silently drops them, which destroys the directory tree. Use ASCII.
            box = str.maketrans({"├": "|", "└": "`", "─": "-", "│": "|"})
            buf = [b.translate(box) for b in buf]
            out += [r"\begin{code}"] + buf + [r"\end{code}"]
            continue

        if re.match(r"^\s*\|", ln):
            close_list()
            rows = []
            while i < len(lines) and re.match(r"^\s*\|", lines[i]):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            if len(rows) >= 2:
                out += table(rows)
            continue

        if re.match(r"^---+\s*$", ln):
            close_list()
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            close_list()
            lvl, txt = len(m.group(1)), inline(m.group(2))
            if lvl == 1:
                out.append(r"\begin{center}{\LARGE\bfseries\color{ink} " + txt + r"}\end{center}")
            elif lvl == 2:
                # keep a section heading from stranding itself at the foot of a page
                out.append(r"\needspace{5\baselineskip}")
                out.append(r"\section*{" + txt + "}")
            else:
                out.append(r"\subsection*{" + txt + "}")
            i += 1
            continue

        m = re.match(r"^\s*[\*\-]\s+(.*)$", ln)
        if m:
            if list_mode != "itemize":
                close_list()
                out.append(r"\begin{itemize}")
                list_mode = "itemize"
            out.append(r"\item " + inline(m.group(1)))
            i += 1
            continue

        m = re.match(r"^\s*\d+\.\s+(.*)$", ln)
        if m:
            if list_mode != "enumerate":
                close_list()
                out.append(r"\begin{enumerate}")
                list_mode = "enumerate"
            out.append(r"\item " + inline(m.group(1)))
            i += 1
            continue

        if not ln.strip():
            close_list()
            out.append("")
        else:
            close_list()
            out.append(inline(ln))
        i += 1

    close_list()
    # collapse runs of blank lines so \parskip does not stack up
    collapsed = []
    for ln in out:
        if ln == "" and collapsed and collapsed[-1] == "":
            continue
        collapsed.append(ln)
    return "\n".join(collapsed)


def main():
    md = open(SRC, encoding="utf-8").read()
    tex = PREAMBLE + convert(md) + "\n\\end{document}\n"
    tex_path = os.path.join(OUTDIR, STEM + ".tex")
    open(tex_path, "w", encoding="utf-8").write(tex)

    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", STEM + ".tex"],
                           cwd=OUTDIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pdf = os.path.join(OUTDIR, STEM + ".pdf")
    if not os.path.exists(pdf):
        print("pdflatex failed; see report/README.log", file=sys.stderr)
        sys.exit(1)
    for ext in (".aux", ".log", ".out", ".tex"):
        p = os.path.join(OUTDIR, STEM + ext)
        if os.path.exists(p):
            os.remove(p)
    print(f"wrote {pdf} ({os.path.getsize(pdf) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
