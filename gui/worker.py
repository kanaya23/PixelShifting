"""
Worker — QThread wrapper around the OptimizerEngine.

Runs the optimization in a background thread and emits Qt signals
for progress updates and completion, keeping the GUI responsive.
"""

import torch
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

from core.device_manager import DeviceManager
from core.optimizer_engine import OptimizerEngine
from utils.image_utils import tensor_to_qimage


class OptimizationWorker(QThread):
    """
    Background thread running the pixel-drift optimization.

    Signals:
        progress(int, int, QImage, dict):
            step, total_steps, preview_image, loss_dict
        finished(QImage):
            Final warped result image.
        error(str):
            Error message if something goes wrong.
    """

    progress = pyqtSignal(int, int, object, dict)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        params: dict,
        parent=None,
    ):
        super().__init__(parent)
        self.source = source
        self.target = target
        self.params = params
        self.engine = None

    def run(self):
        """Execute optimization in background thread."""
        try:
            dm = DeviceManager()

            def on_progress(step, total, warped_tensor, loss_dict):
                qimg = tensor_to_qimage(warped_tensor)
                self.progress.emit(step, total, qimg, loss_dict)

            def on_finished(final_tensor):
                qimg = tensor_to_qimage(final_tensor)
                self.finished.emit(qimg)

            self.engine = OptimizerEngine(
                source=self.source,
                target=self.target,
                device_manager=dm,
                lr=self.params.get("lr", 0.01),
                iterations=self.params.get("iterations", 1000),
                w_sinkhorn=self.params.get("w_sinkhorn", 1.0),
                w_perceptual=self.params.get("w_perceptual", 1.0),
                w_tv=self.params.get("w_tv", 0.1),
                sampling_mode=self.params.get("sampling_mode", "physical"),
                dist_mode=self.params.get("dist_mode", "swd"),
                update_interval=max(1, self.params.get("iterations", 1000) // 100),
                on_progress=on_progress,
                on_finished=on_finished,
            )

            self.engine.run()

        except Exception as e:
            self.error.emit(str(e))

    def pause(self):
        if self.engine:
            self.engine.pause()

    def resume(self):
        if self.engine:
            self.engine.resume()

    def stop(self):
        if self.engine:
            self.engine.stop()
