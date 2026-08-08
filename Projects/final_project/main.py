import os
import argparse
import shutil
import subprocess
from src.run_experiments import run_all_experiments
from src.evaluate import evaluate_all_models
from src.visualize import plot_learning_curves, plot_confusion_matrices
from src.generate_latex_results import generate_latex_results

def compile_latex(report_dir, presentation_dir):
    """
    Compiles report.tex using pdflatex and bibtex, then builds the slide deck
    with python-pptx and converts it to PDF via LibreOffice.
    Runs commands within their respective directories to ensure paths are correct.
    """
    print("\n" + "="*60)
    print("COMPILING REPORT AND PRESENTATION")
    print("="*60)
    
    # 1. Compile Report
    print("1. Compiling report/report.tex...")
    try:
        # Run pdflatex first pass
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "report.tex"],
            cwd=report_dir, check=True, stdout=subprocess.DEVNULL
        )
        # Run bibtex to link citations
        subprocess.run(
            ["bibtex", "report"],
            cwd=report_dir, check=True, stdout=subprocess.DEVNULL
        )
        # Run pdflatex twice more to resolve cross-references and citations
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "report.tex"],
            cwd=report_dir, check=True, stdout=subprocess.DEVNULL
        )
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "report.tex"],
            cwd=report_dir, check=True, stdout=subprocess.DEVNULL
        )
        print("   Report compiled successfully! Output: report/report.pdf")
    except subprocess.CalledProcessError as e:
        print(f"   Error compiling report.tex: {e}")
        
    # 2. Build Presentation Slides
    print("\n2. Building presentation/presentation.pptx...")
    try:
        subprocess.run(
            ["python3", "make_pptx.py"],
            cwd=presentation_dir, check=True, stdout=subprocess.DEVNULL
        )
        print("   Slides built successfully! Output: presentation/presentation.pptx")
    except subprocess.CalledProcessError as e:
        print(f"   Error building presentation.pptx: {e}")
        return

    # 3. Convert the deck to PDF (LibreOffice headless; skipped if unavailable)
    print("\n3. Converting presentation.pptx to PDF...")
    if shutil.which("soffice") is None:
        print("   Skipped: LibreOffice (soffice) not found on PATH.")
    else:
        try:
            subprocess.run(
                ["soffice", "--headless", "--convert-to", "pdf", "presentation.pptx"],
                cwd=presentation_dir, check=True, stdout=subprocess.DEVNULL
            )
            print("   Converted successfully! Output: presentation/presentation.pdf")
        except subprocess.CalledProcessError as e:
            print(f"   Error converting presentation.pptx: {e}")

    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="CS 5330 Final Project Pipeline")
    parser.add_argument("--run", action="store_true", help="Run the full 3x3 training experiments grid")
    parser.add_argument("--eval", action="store_true", help="Evaluate trained models on test set and generate plots")
    parser.add_argument("--compile", action="store_true", help="Generate results_summary.tex and compile PDF report/slides")
    args = parser.parse_args()
    
    DATA_DIR = "/home/shrirag10/Projects/CS5330/Projects/final_project/data"
    RESULTS_DIR = "/home/shrirag10/Projects/CS5330/Projects/final_project/results"
    REPORT_DIR = "/home/shrirag10/Projects/CS5330/Projects/final_project/report"
    PRESENTATION_DIR = "/home/shrirag10/Projects/CS5330/Projects/final_project/presentation"
    CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    
    if not (args.run or args.eval or args.compile):
        parser.print_help()
        return
        
    if args.run:
        print("Launching training run experiments...")
        run_all_experiments(DATA_DIR, RESULTS_DIR)
        
    if args.eval:
        print("Launching evaluation on test set and plotting curves...")
        evaluate_all_models(DATA_DIR, RESULTS_DIR)
        plot_learning_curves(RESULTS_DIR)
        plot_confusion_matrices(RESULTS_DIR, CLASSES)
        
    if args.compile:
        print("Generating results_summary.tex macros...")
        generate_latex_results(RESULTS_DIR, REPORT_DIR)
        compile_latex(REPORT_DIR, PRESENTATION_DIR)

if __name__ == '__main__':
    main()
