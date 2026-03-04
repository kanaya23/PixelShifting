"""
Main Window — PixelShift application window.

Compact layout using QSplitter between image area and controls.
Designed to fit on smaller screens (≥ 900×550).
"""

import os

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QFileDialog, QMessageBox, QLabel, QPushButton,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from gui.image_panel import ImagePanel
from gui.controls_panel import ControlsPanel
from gui.worker import OptimizationWorker
from utils.image_utils import load_image, tensor_to_qimage


# ── Dark Stylesheet ──
DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #12141a;
    color: #d1d5db;
    font-family: 'Segoe UI', 'Inter', sans-serif;
}
QGroupBox { color: #b0b8c8; }
QLabel { color: #d1d5db; }
QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #1e2128;
    border: 1px solid #3a3f4b;
    border-radius: 3px;
    padding: 2px 6px;
    color: #d1d5db;
    font-size: 11px;
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #7c3aed;
}
QComboBox::drop-down { border: none; padding-right: 6px; }
QComboBox QAbstractItemView {
    background-color: #1e2128;
    border: 1px solid #3a3f4b;
    color: #d1d5db;
    selection-background-color: #7c3aed;
}
QSplitter::handle {
    background-color: #3a3f4b;
    height: 3px;
}
QStatusBar {
    background-color: #0d0f14;
    color: #6b7280;
    font-size: 10px;
    border-top: 1px solid #1e2128;
}
"""


class MainWindow(QMainWindow):
    """
    PixelShift main application window.

    Layout (vertical splitter):
        Top:     Source, Preview, Target panels + browse buttons
        Bottom:  Controls panel (compact, horizontal)
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PixelShift — Pixel-Drift Reconstruction")
        self.setMinimumSize(900, 550)
        self.setStyleSheet(DARK_STYLESHEET)

        self._source_path = None
        self._target_path = None
        self._source_tensor = None
        self._target_tensor = None
        self._worker = None
        self._last_result_qimage = None
        self._paused = False

        self._build_ui()
        self._connect_signals()

        self.statusBar().showMessage("Load source and target images to begin")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 4, 6, 2)
        main_layout.setSpacing(4)

        # ── Compact title ──
        title_row = QHBoxLayout()
        title = QLabel("PixelShift")
        title.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #7c3aed;"
        )
        subtitle = QLabel("— Pixel-Drift Reconstruction via Optimal Transport")
        subtitle.setStyleSheet("font-size: 10px; color: #6b7280;")
        title_row.addWidget(title)
        title_row.addWidget(subtitle)
        title_row.addStretch()
        main_layout.addLayout(title_row)

        # ── Vertical splitter: images top, controls bottom ──
        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(4)

        # ── Top: image panels ──
        images_widget = QWidget()
        images_layout = QHBoxLayout(images_widget)
        images_layout.setContentsMargins(0, 0, 0, 0)
        images_layout.setSpacing(6)

        # Source panel + browse
        source_col = QVBoxLayout()
        source_col.setSpacing(2)
        self.source_panel = ImagePanel("Source")
        source_col.addWidget(self.source_panel, stretch=1)
        self.source_browse_btn = QPushButton("📂 Source")
        self.source_browse_btn.setStyleSheet(self._browse_btn_style())
        self.source_browse_btn.setFixedHeight(24)
        self.source_browse_btn.clicked.connect(self._browse_source)
        source_col.addWidget(self.source_browse_btn)
        images_layout.addLayout(source_col, stretch=1)

        # Preview panel (larger)
        preview_col = QVBoxLayout()
        preview_col.setSpacing(2)
        self.preview_panel = ImagePanel("Live Preview")
        self.preview_panel.title_label.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #7c3aed;"
        )
        preview_col.addWidget(self.preview_panel, stretch=1)
        spacer = QLabel("")
        spacer.setFixedHeight(24)
        preview_col.addWidget(spacer)
        images_layout.addLayout(preview_col, stretch=2)

        # Target panel + browse
        target_col = QVBoxLayout()
        target_col.setSpacing(2)
        self.target_panel = ImagePanel("Target")
        target_col.addWidget(self.target_panel, stretch=1)
        self.target_browse_btn = QPushButton("📂 Target")
        self.target_browse_btn.setStyleSheet(self._browse_btn_style())
        self.target_browse_btn.setFixedHeight(24)
        self.target_browse_btn.clicked.connect(self._browse_target)
        target_col.addWidget(self.target_browse_btn)
        images_layout.addLayout(target_col, stretch=1)

        splitter.addWidget(images_widget)

        # ── Bottom: controls panel ──
        self.controls = ControlsPanel()
        splitter.addWidget(self.controls)

        # Give images 65% of the vertical space
        splitter.setSizes([400, 200])

        main_layout.addWidget(splitter, stretch=1)

    def _browse_btn_style(self) -> str:
        return (
            "QPushButton { background-color: #1e2128; color: #b0b8c8; "
            "border: 1px solid #3a3f4b; border-radius: 4px; padding: 3px 8px; "
            "font-size: 11px; } "
            "QPushButton:hover { border-color: #7c3aed; color: #d1d5db; }"
        )

    def _connect_signals(self):
        self.controls.start_clicked.connect(self._on_start)
        self.controls.pause_clicked.connect(self._on_pause)
        self.controls.stop_clicked.connect(self._on_stop)
        self.controls.save_clicked.connect(self._on_save)

    # ── Browse ──

    def _browse_source(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Source Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)"
        )
        if path:
            self._source_path = path
            self.source_panel.set_image_from_path(path)
            self.statusBar().showMessage(f"Source: {os.path.basename(path)}")

    def _browse_target(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Target Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)"
        )
        if path:
            self._target_path = path
            self.target_panel.set_image_from_path(path)
            self.statusBar().showMessage(f"Target: {os.path.basename(path)}")

    # ── Optimization Control ──

    def _on_start(self, params: dict):
        if not self._source_path or not self._target_path:
            QMessageBox.warning(
                self, "Missing Images",
                "Please load both a source and target image before starting."
            )
            return

        resolution = params["resolution"]
        self._source_tensor = load_image(self._source_path, size=resolution)
        self._target_tensor = load_image(self._target_path, size=resolution)

        self.controls.reset_progress()
        self.controls.set_running_state(True)
        self._paused = False
        self.statusBar().showMessage("Optimizing...")

        self._worker = OptimizationWorker(
            source=self._source_tensor,
            target=self._target_tensor,
            params=params,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_pause(self):
        if self._worker:
            if self._paused:
                self._worker.resume()
                self._paused = False
                self.controls.pause_btn.setText("⏸ Pause")
                self.statusBar().showMessage("Resumed")
            else:
                self._worker.pause()
                self._paused = True
                self.controls.pause_btn.setText("▶ Resume")
                self.statusBar().showMessage("Paused")

    def _on_stop(self):
        if self._worker:
            self._worker.stop()
            self.statusBar().showMessage("Stopping...")

    def _on_save(self):
        if self._last_result_qimage is None:
            QMessageBox.information(self, "No Result", "No result to save yet.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Result", "pixelshift_result.png",
            "PNG (*.png);;JPEG (*.jpg);;All Files (*)"
        )
        if path:
            self._last_result_qimage.save(path)
            self.statusBar().showMessage(f"Saved to {os.path.basename(path)}")

    # ── Worker Signals ──

    def _on_progress(self, step: int, total: int, qimage: QImage, losses: dict):
        self.preview_panel.set_image_from_qimage(qimage)
        self.controls.update_progress(step, total, losses)
        self._last_result_qimage = qimage

    def _on_finished(self, qimage: QImage):
        self.preview_panel.set_image_from_qimage(qimage)
        self._last_result_qimage = qimage
        self.controls.set_running_state(False)
        self.controls.save_btn.setEnabled(True)
        self._paused = False
        self.controls.pause_btn.setText("⏸ Pause")
        self.statusBar().showMessage("Done!")

    def _on_error(self, msg: str):
        self.controls.set_running_state(False)
        self._paused = False
        self.controls.pause_btn.setText("⏸ Pause")
        QMessageBox.critical(self, "Error", f"Optimization error:\n{msg}")
        self.statusBar().showMessage(f"Error: {msg}")

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        event.accept()
