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
    from PySide6.QtCore import QObject, QThread, QTimer, Signal as pyqtSignal, Qt
    from PySide6.QtGui import QImage, QPixmap, QFont, QColor
    from PySide6.QtWidgets import (
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
    QT_BINDING = "PySide6"
