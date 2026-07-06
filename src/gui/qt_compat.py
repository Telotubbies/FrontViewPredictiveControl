# Qt compatibility: PyQt5 or PySide6
try:
    from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal, Qt
    from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
    from PyQt5.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QGridLayout,
        QLabel,
        QPushButton,
        QComboBox,
        QSlider,
        QCheckBox,
        QFrame,
        QSplitter,
        QTextEdit,
        QGroupBox,
        QStatusBar,
        QScrollArea,
    )
    QT_BINDING = "PyQt5"
except ImportError:
    QT_BINDING = "PySide6"
