#!/usr/bin/env python3
"""
show_matches.py - visual result viewer for the CS5330 Project 2 match binary.

Usage:
    python3 show_matches.py <target_image> <database_dir> <method> [N] [--embeds <csv>]

Examples:
    python3 show_matches.py olympus/pic.1016.jpg olympus baseline 3
    python3 show_matches.py olympus/pic.0164.jpg olympus dnn 5 --embeds ResNet18_olym.csv
    python3 show_matches.py olympus/pic.0274.jpg olympus custom 5 --embeds ResNet18_olym.csv

The script calls the C++ ./match binary (must be compiled in ./build/),
parses the ranked filenames from stdout, and displays the target image
alongside the top N matches in a single matplotlib figure.

Note: dnn and custom methods require --embeds pointing to ResNet18_olym.csv.
"""

import sys
import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def run_match(target, db_dir, method, n, embeds_csv=None):
    """Run the C++ match binary and return a list of (rank, filename, score)."""
    binary = os.path.join(os.path.dirname(__file__), "build", "match")
    if not os.path.isfile(binary):
        raise FileNotFoundError(f"match binary not found at {binary}. Run 'cmake .. && make' in build/.")

    cmd = [binary, target, db_dir, method, str(n)]

    # dnn and custom require --embeds <csv>; default to one level above db_dir
    if method in ("dnn", "custom"):
        if embeds_csv is None:
            embeds_csv = os.path.join(os.path.dirname(os.path.abspath(db_dir)),
                                      "ResNet18_olym.csv")
        cmd += ["--embeds", embeds_csv]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(1)

    matches = []
    for line in result.stdout.strip().splitlines()[1:]:  # skip header line
        # format: "1. pic.0986.jpg  (score: 14049)"
        parts = line.split(".")
        if len(parts) < 2:
            continue
        try:
            rank = int(parts[0].strip())
            rest = ".".join(parts[1:]).strip()
            fname = rest.split("  ")[0].strip()
            score_str = rest.split("score: ")[-1].rstrip(")")
            score = float(score_str)
            matches.append((rank, fname, score))
        except (ValueError, IndexError):
            continue  # skip malformed or unexpected lines
    return matches


def display(target, db_dir, matches, method, n):
    """Show target + top N matches in a matplotlib grid."""
    total = len(matches) + 1  # target + matches
    cols = min(total, 6)
    rows = (total + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    fig.suptitle(f"Top {n} matches for: {os.path.basename(target)}  [method: {method}]",
                 fontsize=13, fontweight="bold")

    # flatten axes for easy indexing
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    # target image
    ax = axes[0]
    img = mpimg.imread(target)
    ax.imshow(img)
    ax.set_title("TARGET", fontsize=9, fontweight="bold", color="steelblue")
    ax.axis("off")

    # matches
    for i, (rank, fname, score) in enumerate(matches):
        ax = axes[i + 1]
        img_path = os.path.join(db_dir, fname)
        if os.path.isfile(img_path):
            img = mpimg.imread(img_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "not found", ha="center", va="center")
        ax.set_title(f"#{rank}  {fname}\nscore: {score:.4f}", fontsize=7)
        ax.axis("off")

    # hide unused axes
    for ax in axes[len(matches) + 1:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    target  = sys.argv[1]
    db_dir  = sys.argv[2]
    method  = sys.argv[3]
    n       = 5
    embeds  = None

    i = 4
    while i < len(sys.argv):
        if sys.argv[i] == "--embeds" and i + 1 < len(sys.argv):
            embeds = sys.argv[i + 1]
            i += 2
        else:
            try:
                n = int(sys.argv[i])
            except ValueError:
                pass
            i += 1

    if not os.path.isfile(target):
        print(f"Error: target image not found: {target}")
        sys.exit(1)

    matches = run_match(target, db_dir, method, n, embeds)
    display(target, db_dir, matches, method, n)


if __name__ == "__main__":
    main()
