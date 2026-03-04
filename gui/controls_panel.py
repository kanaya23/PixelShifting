"""
Controls Panel — Compact parameter controls, buttons, progress, and loss chart.

Horizontal layout: parameters + buttons on the left, loss chart on the right.
Parameters use a 3-column grid for minimal vertical footprint.
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
        self.fig = Figure(figsize=(4, 2), dpi=90)
        self.fig.patch.set_facecolor("#1a1d23")
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
        self._style_axes()

        self.loss_history = {"sinkhorn": [], "perceptual": [], "tv": [], "total": []}
        self.step_history = []

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(100)
        self.setMinimumWidth(200)

    def _style_axes(self):
        ax = self.ax
        ax.set_facecolor("#1a1d23")
        ax.tick_params(colors="#6b7280", labelsize=7)
        ax.spines["bottom"].set_color("#3a3f4b")
        ax.spines["left"].set_color("#3a3f4b")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Step", color="#6b7280", fontsize=8)
        ax.set_ylabel("Loss", color="#6b7280", fontsize=8)

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
                    linewidth=1.2,
                    label=key,
                    alpha=0.9,
                )

        self.ax.legend(
            fontsize=6,
            loc="upper right",
            facecolor="#1a1d23",
            edgecolor="#3a3f4b",
            labelcolor="#9ca3af",
        )
        self.fig.tight_layout(pad=0.5)
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
    Compact panel: parameters + buttons on the left, loss chart on the right.

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
        # ── Main horizontal layout: controls left, chart right ──
        outer = QHBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(8)

        # ── Left side: params + buttons + progress ──
        left = QVBoxLayout()
        left.setSpacing(4)

        # Parameters in a compact grid
        params_group = QGroupBox("Parameters")
        params_group.setStyleSheet(
            "QGroupBox { color: #b0b8c8; font-weight: bold; font-size: 11px; "
            "border: 1px solid #3a3f4b; border-radius: 4px; "
            "margin-top: 6px; padding-top: 12px; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )
        grid = QGridLayout()
        grid.setSpacing(3)
        grid.setContentsMargins(4, 4, 4, 4)

        # Row 0: Resolution | Sampling | Dist Loss
        grid.addWidget(self._label("Res"), 0, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["128", "256", "512", "1024"])
        self.resolution_combo.setCurrentText("256")
        self.resolution_combo.setFixedHeight(22)
        grid.addWidget(self.resolution_combo, 0, 1)

        grid.addWidget(self._label("Sample"), 0, 2)
        self.sampling_combo = QComboBox()
        self.sampling_combo.addItems(["bilinear", "nearest"])
        self.sampling_combo.setFixedHeight(22)
        grid.addWidget(self.sampling_combo, 0, 3)

        grid.addWidget(self._label("Dist"), 0, 4)
        self.dist_combo = QComboBox()
        self.dist_combo.addItems(["SWD (Fast)", "Sinkhorn"])
        self.dist_combo.setFixedHeight(22)
        grid.addWidget(self.dist_combo, 0, 5)

        # Row 1: LR | Iterations
        grid.addWidget(self._label("LR"), 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setFixedHeight(22)
        grid.addWidget(self.lr_spin, 1, 1)

        grid.addWidget(self._label("Iters"), 1, 2)
        self.iters_spin = QSpinBox()
        self.iters_spin.setRange(100, 10000)
        self.iters_spin.setValue(1000)
        self.iters_spin.setSingleStep(100)
        self.iters_spin.setFixedHeight(22)
        grid.addWidget(self.iters_spin, 1, 3)

        # Row 2: Weights
        grid.addWidget(self._label("W·Dist"), 2, 0)
        self.w_sink_spin = QDoubleSpinBox()
        self.w_sink_spin.setRange(0.0, 10.0)
        self.w_sink_spin.setValue(1.0)
        self.w_sink_spin.setSingleStep(0.1)
        self.w_sink_spin.setDecimals(2)
        self.w_sink_spin.setFixedHeight(22)
        grid.addWidget(self.w_sink_spin, 2, 1)

        grid.addWidget(self._label("W·Perc"), 2, 2)
        self.w_perc_spin = QDoubleSpinBox()
        self.w_perc_spin.setRange(0.0, 10.0)
        self.w_perc_spin.setValue(1.0)
        self.w_perc_spin.setSingleStep(0.1)
        self.w_perc_spin.setDecimals(2)
        self.w_perc_spin.setFixedHeight(22)
        grid.addWidget(self.w_perc_spin, 2, 3)

        grid.addWidget(self._label("W·TV"), 2, 4)
        self.w_tv_spin = QDoubleSpinBox()
        self.w_tv_spin.setRange(0.0, 5.0)
        self.w_tv_spin.setValue(0.1)
        self.w_tv_spin.setSingleStep(0.01)
        self.w_tv_spin.setDecimals(3)
        self.w_tv_spin.setFixedHeight(22)
        grid.addWidget(self.w_tv_spin, 2, 5)

        params_group.setLayout(grid)
        left.addWidget(params_group)

        # ── Buttons row ──
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        self.start_btn = QPushButton("▶ Start")
        self.start_btn.setStyleSheet(self._btn_style("#7c3aed", "#6d28d9"))
        self.start_btn.clicked.connect(self._on_start)
        btn_layout.addWidget(self.start_btn)

        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.setStyleSheet(self._btn_style("#f59e0b", "#d97706"))
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.pause_clicked.emit)
        btn_layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setStyleSheet(self._btn_style("#ef4444", "#dc2626"))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        btn_layout.addWidget(self.stop_btn)

        self.save_btn = QPushButton("💾 Save")
        self.save_btn.setStyleSheet(self._btn_style("#10b981", "#059669"))
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_clicked.emit)
        btn_layout.addWidget(self.save_btn)

        left.addLayout(btn_layout)

        # ── Progress bar ──
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Ready")
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #3a3f4b; border-radius: 3px; "
            "background-color: #1a1d23; text-align: center; color: #b0b8c8; "
            "font-size: 10px; } "
            "QProgressBar::chunk { background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0, stop:0 #7c3aed, stop:1 #06b6d4); "
            "border-radius: 2px; }"
        )
        left.addWidget(self.progress_bar)

        # ── Loss readout ──
        self.loss_label = QLabel("Losses: —")
        self.loss_label.setStyleSheet(
            "color: #9ca3af; font-size: 10px; font-family: monospace;"
        )
        self.loss_label.setAlignment(Qt.AlignCenter)
        left.addWidget(self.loss_label)

        outer.addLayout(left, stretch=3)

        # ── Right side: Loss Chart ──
        self.loss_chart = LossChart(self)
        outer.addWidget(self.loss_chart, stretch=2)

    def _label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #9ca3af; font-size: 10px;")
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return lbl

    def _btn_style(self, bg: str, hover: str) -> str:
        return (
            f"QPushButton {{ background-color: {bg}; color: white; "
            f"border: none; border-radius: 4px; padding: 5px 8px; "
            f"font-weight: bold; font-size: 11px; }} "
            f"QPushButton:hover {{ background-color: {hover}; }} "
            f"QPushButton:disabled {{ background-color: #374151; color: #6b7280; }}"
        )

    def _on_start(self):
        """Gather parameters and emit start signal."""
        dist_mode = "swd" if self.dist_combo.currentIndex() == 0 else "sinkhorn"
        params = {
            "resolution": int(self.resolution_combo.currentText()),
            "sampling_mode": self.sampling_combo.currentText(),
            "dist_mode": dist_mode,
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
        for w in (self.resolution_combo, self.sampling_combo, self.dist_combo,
                  self.lr_spin, self.iters_spin, self.w_sink_spin,
                  self.w_perc_spin, self.w_tv_spin):
            w.setEnabled(not running)

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
