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
    try:
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
    except ImportError:
        QT_BINDING = None

        class QObject:  # type: ignore[no-redef]
            pass

        class QThread:  # type: ignore[no-redef]
            pass

        class QTimer:  # type: ignore[no-redef]
            pass

        def pyqtSignal(*_args, **_kwargs):  # type: ignore[no-redef]
            class _DummySignal:
                def connect(self, *_a, **_k): pass
                def emit(self, *_a, **_k): pass
            return _DummySignal()

        class Qt:  # type: ignore[no-redef]
            AlignCenter = 0

        QImage = QPixmap = QFont = QColor = None  # type: ignore[assignment]
        QApplication = QMainWindow = QWidget = QVBoxLayout = QHBoxLayout = None  # type: ignore[assignment]
        QGridLayout = QLabel = QPushButton = QComboBox = QSlider = QCheckBox = None  # type: ignore[assignment]
        QFrame = QSplitter = QTextEdit = QGroupBox = QStatusBar = QScrollArea = None  # type: ignore[assignment]
