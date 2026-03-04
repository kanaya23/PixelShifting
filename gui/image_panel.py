"""
Image Panel — Reusable QLabel-based widget for image display.

Displays images with automatic aspect-ratio-preserving scaling,
a title label, and optional drag-and-drop border styling.
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt, QSize


class ImagePanel(QWidget):
    """
    Displays a single image with a title label.

    The image is scaled to fit the panel while preserving aspect ratio.
    Shows a placeholder when no image is loaded.
    """

    def __init__(self, title: str = "Image", parent=None):
        super().__init__(parent)
        self._title = title
        self._pixmap = None

        self.setMinimumSize(200, 200)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Title label
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #b0b8c8;"
        )
        layout.addWidget(self.title_label)

        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(180, 180)
        self.image_label.setStyleSheet(
            "border: 2px dashed #3a3f4b; border-radius: 8px; "
            "background-color: #1e2128;"
        )
        self.image_label.setText("No image loaded")
        layout.addWidget(self.image_label, stretch=1)

    def set_image_from_qimage(self, qimage: QImage):
        """Set the display image from a QImage."""
        self._pixmap = QPixmap.fromImage(qimage)
        self._update_display()

    def set_image_from_pixmap(self, pixmap: QPixmap):
        """Set the display image from a QPixmap."""
        self._pixmap = pixmap
        self._update_display()

    def set_image_from_path(self, path: str):
        """Load and display an image from a file path."""
        self._pixmap = QPixmap(path)
        if self._pixmap.isNull():
            self.image_label.setText("Failed to load image")
            self._pixmap = None
        else:
            self._update_display()

    def clear(self):
        """Clear the displayed image."""
        self._pixmap = None
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("No image loaded")

    def _update_display(self):
        """Scale and display the current pixmap."""
        if self._pixmap is None:
            return
        label_size = self.image_label.size()
        scaled = self._pixmap.scaled(
            label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)
        self.image_label.setText("")

    def resizeEvent(self, event):
        """Re-scale image on panel resize."""
        super().resizeEvent(event)
        if self._pixmap is not None:
            self._update_display()
