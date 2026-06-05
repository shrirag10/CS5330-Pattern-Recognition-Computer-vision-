#!/usr/bin/env python3
"""
gui.py -- PyQt5 GUI for the CS5330 Project 2 CBIR system.

Usage:
    python3 gui.py

Requirements:
    pip install PyQt5
    Build the C++ match binary first: cd build && cmake .. && make

The GUI lets you:
  1. Browse for a target image
  2. Browse for the database directory (default: olympus/)
  3. Choose a method and N
  4. Click "Find Matches" to run the C++ binary and display results
"""

import sys
import os
import subprocess

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QFileDialog,
    QScrollArea, QFrame, QLineEdit, QGroupBox
)
from PyQt5.QtGui import QPixmap, QFont, QColor, QPalette
from PyQt5.QtCore import Qt, QThread, pyqtSignal


# ── background worker so the UI doesn't freeze during matching ─────────────
class MatchWorker(QThread):
    finished = pyqtSignal(list)   # emits list of (rank, filename, score)
    error    = pyqtSignal(str)

    def __init__(self, binary, target, db_dir, method, n):
        super().__init__()
        self.binary  = binary
        self.target  = target
        self.db_dir  = db_dir
        self.method  = method
        self.n       = n

    def run(self):
        cmd = [self.binary, self.target, self.db_dir, self.method, str(self.n)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except Exception as e:
            self.error.emit(str(e))
            return

        if result.returncode != 0:
            self.error.emit(result.stderr.strip() or "match binary returned non-zero exit code")
            return

        matches = []
        for line in result.stdout.strip().splitlines()[1:]:
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
                continue
        self.finished.emit(matches)


# ── image card widget ───────────────────────────────────────────────────────
class ImageCard(QFrame):
    THUMB = 200  # thumbnail size in pixels

    def __init__(self, img_path, label_text, score_text="", is_target=False):
        super().__init__()
        self.setFixedWidth(self.THUMB + 16)
        self.setFrameShape(QFrame.StyledPanel)

        if is_target:
            self.setStyleSheet("QFrame { border: 3px solid #4A90D9; border-radius: 8px; background: #1e2a38; }")
        else:
            self.setStyleSheet("QFrame { border: 1px solid #3a3a3a; border-radius: 8px; background: #1e1e1e; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # thumbnail
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignCenter)
        if os.path.isfile(img_path):
            pix = QPixmap(img_path).scaled(
                self.THUMB, self.THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img_label.setPixmap(pix)
        else:
            img_label.setText("image\nnot found")
            img_label.setStyleSheet("color: #888;")
        img_label.setFixedHeight(self.THUMB)
        layout.addWidget(img_label)

        # filename label
        lbl = QLabel(label_text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setWordWrap(True)
        font = QFont()
        font.setPointSize(8)
        font.setBold(is_target)
        lbl.setFont(font)
        lbl.setStyleSheet("color: #4A90D9;" if is_target else "color: #cccccc;")
        layout.addWidget(lbl)

        # score label
        if score_text:
            slbl = QLabel(score_text)
            slbl.setAlignment(Qt.AlignCenter)
            sf = QFont()
            sf.setPointSize(7)
            slbl.setFont(sf)
            slbl.setStyleSheet("color: #aaaaaa;")
            layout.addWidget(slbl)


# ── main window ─────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):

    METHODS = [
        "baseline",
        "histogram",
        "multihistogram",
        "multihistogram-weighted",
        "texture",
        "dnn",
        "custom",
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CS5330 CBIR - Content-Based Image Retrieval")
        self.setMinimumSize(1100, 680)
        self._apply_dark_theme()

        # locate the match binary relative to this script
        self.binary = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "build", "match")

        self._build_ui()
        self.statusBar().showMessage("Ready: select a target image and click Find Matches.")

    # ── dark theme ────────────────────────────────────────────────────────
    def _apply_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window,          QColor("#121212"))
        palette.setColor(QPalette.WindowText,      QColor("#e0e0e0"))
        palette.setColor(QPalette.Base,            QColor("#1e1e1e"))
        palette.setColor(QPalette.AlternateBase,   QColor("#2a2a2a"))
        palette.setColor(QPalette.Text,            QColor("#e0e0e0"))
        palette.setColor(QPalette.Button,          QColor("#2a2a2a"))
        palette.setColor(QPalette.ButtonText,      QColor("#e0e0e0"))
        palette.setColor(QPalette.Highlight,       QColor("#4A90D9"))
        palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        self.setPalette(palette)
        self.setStyleSheet("""
            QMainWindow  { background: #121212; }
            QGroupBox    { border: 1px solid #3a3a3a; border-radius: 6px;
                           margin-top: 8px; color: #aaaaaa; font-size: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; }
            QLabel       { color: #e0e0e0; }
            QLineEdit    { background: #1e1e1e; border: 1px solid #3a3a3a;
                           border-radius: 4px; color: #e0e0e0; padding: 4px; }
            QComboBox    { background: #1e1e1e; border: 1px solid #3a3a3a;
                           border-radius: 4px; color: #e0e0e0; padding: 4px; }
            QSpinBox     { background: #1e1e1e; border: 1px solid #3a3a3a;
                           border-radius: 4px; color: #e0e0e0; padding: 4px; }
            QPushButton  { background: #2a2a2a; border: 1px solid #3a3a3a;
                           border-radius: 4px; color: #e0e0e0; padding: 6px 14px; }
            QPushButton:hover   { background: #3a3a3a; }
            QPushButton#run_btn { background: #4A90D9; border: none; color: white;
                                  font-weight: bold; padding: 8px 20px; }
            QPushButton#run_btn:hover    { background: #357abd; }
            QPushButton#run_btn:disabled { background: #2a2a2a; color: #555; }
            QScrollArea  { border: none; background: #121212; }
            QStatusBar   { color: #888888; font-size: 10px; }
        """)

    # ── UI construction ───────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 8)
        root.setSpacing(10)

        # ── title bar ────────────────────────────────────────────────────
        title = QLabel("Content-Based Image Retrieval")
        tf = QFont()
        tf.setPointSize(16)
        tf.setBold(True)
        title.setFont(tf)
        title.setStyleSheet("color: #4A90D9;")
        root.addWidget(title)

        sub = QLabel("CS5330 Computer Vision · Project 2")
        sf = QFont()
        sf.setPointSize(9)
        sub.setFont(sf)
        sub.setStyleSheet("color: #666666;")
        root.addWidget(sub)

        # ── controls row ──────────────────────────────────────────────────
        ctrl_grp = QGroupBox("Query Settings")
        ctrl_lay = QHBoxLayout(ctrl_grp)
        ctrl_lay.setSpacing(12)

        # target image
        target_grp = QGroupBox("Target Image")
        tg_lay = QHBoxLayout(target_grp)
        self.target_edit = QLineEdit()
        self.target_edit.setPlaceholderText("Select target image…")
        tg_lay.addWidget(self.target_edit)
        btn_target = QPushButton("Browse…")
        btn_target.clicked.connect(self._pick_target)
        tg_lay.addWidget(btn_target)
        ctrl_lay.addWidget(target_grp, 3)

        # database directory
        db_grp = QGroupBox("Database Directory")
        db_lay = QHBoxLayout(db_grp)
        self.db_edit = QLineEdit()
        default_db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "olympus")
        self.db_edit.setText(default_db if os.path.isdir(default_db) else "")
        db_lay.addWidget(self.db_edit)
        btn_db = QPushButton("Browse…")
        btn_db.clicked.connect(self._pick_db)
        db_lay.addWidget(btn_db)
        ctrl_lay.addWidget(db_grp, 2)

        # method
        meth_grp = QGroupBox("Method")
        meth_lay = QVBoxLayout(meth_grp)
        self.method_combo = QComboBox()
        self.method_combo.addItems(self.METHODS)
        self.method_combo.setCurrentText("dnn")
        meth_lay.addWidget(self.method_combo)
        ctrl_lay.addWidget(meth_grp, 1)

        # N
        n_grp = QGroupBox("Results (N)")
        n_lay = QVBoxLayout(n_grp)
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 20)
        self.n_spin.setValue(5)
        n_lay.addWidget(self.n_spin)
        ctrl_lay.addWidget(n_grp)

        # run button
        self.run_btn = QPushButton("Find Matches")
        self.run_btn.setObjectName("run_btn")
        self.run_btn.setFixedHeight(48)
        self.run_btn.clicked.connect(self._run_query)
        ctrl_lay.addWidget(self.run_btn)

        root.addWidget(ctrl_grp)

        # ── results scroll area ───────────────────────────────────────────
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.results_widget = QWidget()
        self.results_layout = QHBoxLayout(self.results_widget)
        self.results_layout.setAlignment(Qt.AlignLeft)
        self.results_layout.setSpacing(10)

        placeholder = QLabel("Results will appear here after you run a query.")
        placeholder.setStyleSheet("color: #444444; font-size: 13px;")
        placeholder.setAlignment(Qt.AlignCenter)
        self.results_layout.addWidget(placeholder)

        self.scroll.setWidget(self.results_widget)
        root.addWidget(self.scroll, 1)

    # ── helpers ───────────────────────────────────────────────────────────
    def _pick_target(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Target Image", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if path:
            self.target_edit.setText(path)

    def _pick_db(self):
        path = QFileDialog.getExistingDirectory(self, "Select Database Directory")
        if path:
            self.db_edit.setText(path)

    def _clear_results(self):
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _run_query(self):
        target = self.target_edit.text().strip()
        db_dir = self.db_edit.text().strip()
        method = self.method_combo.currentText()
        n      = self.n_spin.value()

        if not target or not os.path.isfile(target):
            self.statusBar().showMessage("Error: please select a valid target image.")
            return
        if not db_dir or not os.path.isdir(db_dir):
            self.statusBar().showMessage("Error: please select a valid database directory.")
            return
        if not os.path.isfile(self.binary):
            self.statusBar().showMessage(f"Error: match binary not found at {self.binary}")
            return

        self.run_btn.setEnabled(False)
        self.statusBar().showMessage(f"Running {method} on {os.path.basename(target)}…")
        self._clear_results()

        self._worker = MatchWorker(self.binary, target, db_dir, method, n)
        self._worker.finished.connect(self._show_results)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _show_results(self, matches):
        self.run_btn.setEnabled(True)
        self._clear_results()

        target = self.target_edit.text().strip()
        db_dir = self.db_edit.text().strip()

        # target card
        target_card = ImageCard(
            target,
            f"TARGET\n{os.path.basename(target)}",
            is_target=True
        )
        self.results_layout.addWidget(target_card)

        # separator
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #3a3a3a;")
        self.results_layout.addWidget(sep)

        # match cards
        for rank, fname, score in matches:
            img_path = os.path.join(db_dir, fname)
            card = ImageCard(img_path, fname, f"#{rank}  score: {score:.4f}")
            self.results_layout.addWidget(card)

        self.results_layout.addStretch()

        method = self.method_combo.currentText()
        self.statusBar().showMessage(
            f"Showing top {len(matches)} matches for {os.path.basename(target)} "
            f"[method: {method}]"
        )

    def _on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.statusBar().showMessage(f"Error: {msg}")


# ── entry point ─────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CS5330 CBIR")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
