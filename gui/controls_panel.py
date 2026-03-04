"""
Controls Panel — Parameter controls, buttons, progress bar, and loss chart.

Provides all adjustable parameters (LR, iterations, weights, resolution,
sampling mode) as well as Start/Pause/Stop/Save buttons, a progress bar,
loss readouts, and an embedded matplotlib loss curve chart.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QProgressBar, QComboBox,
    QDoubleSpinBox, QSpinBox, QGroupBox, QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class LossChart(FigureCanvas):
    """Embedded matplotlib chart showing live loss curves."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 2.2), dpi=100)
        self.fig.patch.set_facecolor("#1a1d23")
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
        self._style_axes()

        self.loss_history = {"sinkhorn": [], "perceptual": [], "tv": [], "total": []}
        self.step_history = []

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(160)

    def _style_axes(self):
        ax = self.ax
        ax.set_facecolor("#1a1d23")
        ax.tick_params(colors="#6b7280", labelsize=8)
        ax.spines["bottom"].set_color("#3a3f4b")
        ax.spines["left"].set_color("#3a3f4b")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Step", color="#6b7280", fontsize=9)
        ax.set_ylabel("Loss", color="#6b7280", fontsize=9)

    def update_chart(self, step: int, losses: dict):
        """Add a data point and redraw."""
        self.step_history.append(step)
        for key in self.loss_history:
            self.loss_history[key].append(losses.get(key, 0.0))

        self.ax.clear()
        self._style_axes()

        colors = {
            "total": "#7c3aed",
            "sinkhorn": "#06b6d4",
            "perceptual": "#f59e0b",
            "tv": "#10b981",
        }
        for key, color in colors.items():
            if self.loss_history[key]:
                self.ax.plot(
                    self.step_history,
                    self.loss_history[key],
                    color=color,
                    linewidth=1.4,
                    label=key,
                    alpha=0.9,
                )

        self.ax.legend(
            fontsize=7,
            loc="upper right",
            facecolor="#1a1d23",
            edgecolor="#3a3f4b",
            labelcolor="#9ca3af",
        )
        self.fig.tight_layout(pad=0.8)
        self.draw()

    def reset(self):
        """Clear all history and redraw."""
        self.loss_history = {"sinkhorn": [], "perceptual": [], "tv": [], "total": []}
        self.step_history = []
        self.ax.clear()
        self._style_axes()
        self.draw()


class ControlsPanel(QWidget):
    """
    Panel containing all parameter controls, action buttons,
    progress display, and embedded loss chart.

    Signals:
        start_clicked(dict):  Emitted with all current parameter values.
        pause_clicked():      Emitted when Pause is clicked.
        stop_clicked():       Emitted when Stop is clicked.
        save_clicked():       Emitted when Save is clicked.
    """

    start_clicked = pyqtSignal(dict)
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    save_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Parameters Group ──
        params_group = QGroupBox("Parameters")
        params_group.setStyleSheet(
            "QGroupBox { color: #b0b8c8; font-weight: bold; border: 1px solid #3a3f4b; "
            "border-radius: 6px; margin-top: 8px; padding-top: 14px; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; }"
        )
        grid = QGridLayout()
        grid.setSpacing(6)

        row = 0

        # Resolution
        grid.addWidget(self._label("Resolution"), row, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["128", "256", "512", "1024"])
        self.resolution_combo.setCurrentText("256")
        grid.addWidget(self.resolution_combo, row, 1)
        row += 1

        # Sampling Mode
        grid.addWidget(self._label("Sampling"), row, 0)
        self.sampling_combo = QComboBox()
        self.sampling_combo.addItems(["bilinear", "nearest"])
        grid.addWidget(self.sampling_combo, row, 1)
        row += 1

        # Learning Rate
        grid.addWidget(self._label("Learning Rate"), row, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setDecimals(4)
        grid.addWidget(self.lr_spin, row, 1)
        row += 1

        # Iterations
        grid.addWidget(self._label("Iterations"), row, 0)
        self.iters_spin = QSpinBox()
        self.iters_spin.setRange(100, 10000)
        self.iters_spin.setValue(1000)
        self.iters_spin.setSingleStep(100)
        grid.addWidget(self.iters_spin, row, 1)
        row += 1

        # Sinkhorn weight
        grid.addWidget(self._label("W Sinkhorn"), row, 0)
        self.w_sink_spin = QDoubleSpinBox()
        self.w_sink_spin.setRange(0.0, 10.0)
        self.w_sink_spin.setValue(1.0)
        self.w_sink_spin.setSingleStep(0.1)
        self.w_sink_spin.setDecimals(2)
        grid.addWidget(self.w_sink_spin, row, 1)
        row += 1

        # Perceptual weight
        grid.addWidget(self._label("W Perceptual"), row, 0)
        self.w_perc_spin = QDoubleSpinBox()
        self.w_perc_spin.setRange(0.0, 10.0)
        self.w_perc_spin.setValue(1.0)
        self.w_perc_spin.setSingleStep(0.1)
        self.w_perc_spin.setDecimals(2)
        grid.addWidget(self.w_perc_spin, row, 1)
        row += 1

        # TV weight
        grid.addWidget(self._label("W TV"), row, 0)
        self.w_tv_spin = QDoubleSpinBox()
        self.w_tv_spin.setRange(0.0, 5.0)
        self.w_tv_spin.setValue(0.1)
        self.w_tv_spin.setSingleStep(0.01)
        self.w_tv_spin.setDecimals(3)
        grid.addWidget(self.w_tv_spin, row, 1)
        row += 1

        params_group.setLayout(grid)
        layout.addWidget(params_group)

        # ── Action Buttons ──
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(6)

        self.start_btn = QPushButton("▶  Start")
        self.start_btn.setStyleSheet(self._btn_style("#7c3aed", "#6d28d9"))
        self.start_btn.clicked.connect(self._on_start)
        btn_layout.addWidget(self.start_btn)

        self.pause_btn = QPushButton("⏸  Pause")
        self.pause_btn.setStyleSheet(self._btn_style("#f59e0b", "#d97706"))
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.pause_clicked.emit)
        btn_layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("⏹  Stop")
        self.stop_btn.setStyleSheet(self._btn_style("#ef4444", "#dc2626"))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        btn_layout.addWidget(self.stop_btn)

        self.save_btn = QPushButton("💾  Save")
        self.save_btn.setStyleSheet(self._btn_style("#10b981", "#059669"))
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_clicked.emit)
        btn_layout.addWidget(self.save_btn)

        layout.addLayout(btn_layout)

        # ── Progress ──
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Ready")
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #3a3f4b; border-radius: 4px; "
            "background-color: #1a1d23; text-align: center; color: #b0b8c8; "
            "font-size: 11px; height: 20px; } "
            "QProgressBar::chunk { background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0, stop:0 #7c3aed, stop:1 #06b6d4); "
            "border-radius: 3px; }"
        )
        layout.addWidget(self.progress_bar)

        # ── Loss readout ──
        self.loss_label = QLabel("Losses: —")
        self.loss_label.setStyleSheet(
            "color: #9ca3af; font-size: 11px; font-family: monospace;"
        )
        self.loss_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.loss_label)

        # ── Loss Chart ──
        self.loss_chart = LossChart(self)
        layout.addWidget(self.loss_chart)

        layout.addStretch()

    def _label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #9ca3af; font-size: 12px;")
        return lbl

    def _btn_style(self, bg: str, hover: str) -> str:
        return (
            f"QPushButton {{ background-color: {bg}; color: white; "
            f"border: none; border-radius: 6px; padding: 8px 12px; "
            f"font-weight: bold; font-size: 12px; }} "
            f"QPushButton:hover {{ background-color: {hover}; }} "
            f"QPushButton:disabled {{ background-color: #374151; color: #6b7280; }}"
        )

    def _on_start(self):
        """Gather parameters and emit start signal."""
        params = {
            "resolution": int(self.resolution_combo.currentText()),
            "sampling_mode": self.sampling_combo.currentText(),
            "lr": self.lr_spin.value(),
            "iterations": self.iters_spin.value(),
            "w_sinkhorn": self.w_sink_spin.value(),
            "w_perceptual": self.w_perc_spin.value(),
            "w_tv": self.w_tv_spin.value(),
        }
        self.start_clicked.emit(params)

    def set_running_state(self, running: bool):
        """Toggle button enabled states for running vs idle."""
        self.start_btn.setEnabled(not running)
        self.pause_btn.setEnabled(running)
        self.stop_btn.setEnabled(running)
        self.save_btn.setEnabled(not running)
        # Disable parameter changes while running
        self.resolution_combo.setEnabled(not running)
        self.sampling_combo.setEnabled(not running)
        self.lr_spin.setEnabled(not running)
        self.iters_spin.setEnabled(not running)
        self.w_sink_spin.setEnabled(not running)
        self.w_perc_spin.setEnabled(not running)
        self.w_tv_spin.setEnabled(not running)

    def update_progress(self, step: int, total: int, losses: dict):
        """Update progress bar, loss readout, and chart."""
        pct = int(100 * step / total)
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"Step {step}/{total}  ({pct}%)")

        parts = []
        for key in ("sinkhorn", "perceptual", "tv", "total"):
            if key in losses:
                parts.append(f"{key}: {losses[key]:.4f}")
        self.loss_label.setText("  │  ".join(parts))

        self.loss_chart.update_chart(step, losses)

    def reset_progress(self):
        """Reset progress display for a new run."""
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Ready")
        self.loss_label.setText("Losses: —")
        self.loss_chart.reset()
