"""
Main Window — PixelShift application window.

Combines three image panels (source, preview, target), the controls panel,
and wires everything to the background optimization worker.
"""

import os

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QFileDialog, QMessageBox, QLabel, QPushButton, QStatusBar,
)
from PyQt5.QtGui import QPixmap, QIcon, QImage
from PyQt5.QtCore import Qt

from gui.image_panel import ImagePanel
from gui.controls_panel import ControlsPanel
from gui.worker import OptimizationWorker
from utils.image_utils import load_image, tensor_to_qimage, save_image


# ── Dark Stylesheet ──
DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #12141a;
    color: #d1d5db;
    font-family: 'Segoe UI', 'Inter', sans-serif;
}
QGroupBox {
    color: #b0b8c8;
}
QLabel {
    color: #d1d5db;
}
QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #1e2128;
    border: 1px solid #3a3f4b;
    border-radius: 4px;
    padding: 4px 8px;
    color: #d1d5db;
    font-size: 12px;
    min-height: 24px;
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #7c3aed;
}
QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #1e2128;
    border: 1px solid #3a3f4b;
    color: #d1d5db;
    selection-background-color: #7c3aed;
}
QSplitter::handle {
    background-color: #3a3f4b;
    width: 2px;
}
QStatusBar {
    background-color: #0d0f14;
    color: #6b7280;
    font-size: 11px;
    border-top: 1px solid #1e2128;
}
"""


class MainWindow(QMainWindow):
    """
    PixelShift main application window.

    Layout:
        Left:    Source panel + browse button
        Center:  Preview panel (live warped result)
        Right:   Target panel + browse button
        Bottom:  Controls panel (parameters, buttons, chart)
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PixelShift — Pixel-Drift Reconstruction")
        self.setMinimumSize(1100, 750)
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

        # Status bar
        self.statusBar().showMessage("Ready — Load source and target images to begin")

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # ── Title ──
        title = QLabel("PixelShift")
        title.setStyleSheet(
            "font-size: 22px; font-weight: bold; "
            "color: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #7c3aed, stop:1 #06b6d4); "
            "margin-bottom: 4px;"
        )
        title.setAlignment(Qt.AlignCenter)
        subtitle = QLabel("Pixel-Drift Reconstruction via Optimal Transport")
        subtitle.setStyleSheet("font-size: 12px; color: #6b7280; margin-bottom: 8px;")
        subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        # ── Image Panels Row ──
        images_layout = QHBoxLayout()
        images_layout.setSpacing(8)

        # Source panel + browse
        source_col = QVBoxLayout()
        self.source_panel = ImagePanel("Source Image")
        source_col.addWidget(self.source_panel, stretch=1)
        self.source_browse_btn = QPushButton("📂  Browse Source")
        self.source_browse_btn.setStyleSheet(self._browse_btn_style())
        self.source_browse_btn.clicked.connect(self._browse_source)
        source_col.addWidget(self.source_browse_btn)
        images_layout.addLayout(source_col, stretch=1)

        # Preview panel (larger)
        preview_col = QVBoxLayout()
        self.preview_panel = ImagePanel("Live Preview")
        self.preview_panel.title_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #7c3aed;"
        )
        preview_col.addWidget(self.preview_panel, stretch=1)
        preview_spacer = QLabel("")  # spacer to align with browse buttons
        preview_spacer.setFixedHeight(32)
        preview_col.addWidget(preview_spacer)
        images_layout.addLayout(preview_col, stretch=2)

        # Target panel + browse
        target_col = QVBoxLayout()
        self.target_panel = ImagePanel("Target Image")
        target_col.addWidget(self.target_panel, stretch=1)
        self.target_browse_btn = QPushButton("📂  Browse Target")
        self.target_browse_btn.setStyleSheet(self._browse_btn_style())
        self.target_browse_btn.clicked.connect(self._browse_target)
        target_col.addWidget(self.target_browse_btn)
        images_layout.addLayout(target_col, stretch=1)

        main_layout.addLayout(images_layout, stretch=1)

        # ── Controls Panel ──
        self.controls = ControlsPanel()
        main_layout.addWidget(self.controls)

    def _browse_btn_style(self) -> str:
        return (
            "QPushButton { background-color: #1e2128; color: #b0b8c8; "
            "border: 1px solid #3a3f4b; border-radius: 6px; padding: 6px 12px; "
            "font-size: 12px; } "
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

        # Load images at selected resolution
        resolution = params["resolution"]
        self._source_tensor = load_image(self._source_path, size=resolution)
        self._target_tensor = load_image(self._target_path, size=resolution)

        # Reset progress
        self.controls.reset_progress()
        self.controls.set_running_state(True)
        self._paused = False
        self.statusBar().showMessage("Optimizing...")

        # Create and start worker
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
                self.controls.pause_btn.setText("⏸  Pause")
                self.statusBar().showMessage("Resumed")
            else:
                self._worker.pause()
                self._paused = True
                self.controls.pause_btn.setText("▶  Resume")
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
        self.controls.pause_btn.setText("⏸  Pause")
        self.statusBar().showMessage("Optimization complete!")

    def _on_error(self, msg: str):
        self.controls.set_running_state(False)
        self._paused = False
        self.controls.pause_btn.setText("⏸  Pause")
        QMessageBox.critical(self, "Error", f"Optimization error:\n{msg}")
        self.statusBar().showMessage(f"Error: {msg}")

    def closeEvent(self, event):
        """Clean up worker thread on window close."""
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        event.accept()
