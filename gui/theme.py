# Dark theme and styling

DARK_THEME = """
    QMainWindow, QWidget, QDialog { background-color: #121212; }
    QLabel { color: #e0e0e0; }
    QGroupBox { color: #b0b0b0; font-weight: bold; border: 1px solid #333; border-radius: 6px; margin-top: 8px; padding-top: 6px; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
    QPushButton {
        background-color: #2a2a2a; color: #e0e0e0; border: 1px solid #444;
        border-radius: 6px; padding: 8px 14px; min-width: 80px;
    }
    QPushButton:hover { background-color: #353535; border-color: #00bcd4; }
    QPushButton:pressed { background-color: #1e1e1e; }
    QPushButton:disabled { color: #666; background-color: #1a1a1a; }
    QPushButton#startBtn { background-color: #1b5e20; }
    QPushButton#startBtn:hover { background-color: #2e7d32; }
    QPushButton#stopBtn { background-color: #b71c1c; }
    QPushButton#stopBtn:hover { background-color: #c62828; }
    QPushButton#emergencyBtn { background-color: #b71c1c; color: white; font-weight: bold; }
    QPushButton#emergencyBtn:hover { background-color: #d32f2f; }
    QComboBox {
        background-color: #2a2a2a; color: #e0e0e0; border: 1px solid #444;
        border-radius: 4px; padding: 4px 8px; min-width: 100px;
    }
    QComboBox:hover { border-color: #00bcd4; }
    QSlider::groove:horizontal { height: 6px; background: #333; border-radius: 3px; }
    QSlider::handle:horizontal { width: 14px; margin: -4px 0; background: #00bcd4; border-radius: 7px; }
    QSlider::sub-page:horizontal { background: #00bcd4; border-radius: 3px; }
    QCheckBox { color: #e0e0e0; spacing: 6px; }
    QCheckBox::indicator { width: 18px; height: 18px; border-radius: 4px; border: 2px solid #555; background: #2a2a2a; }
    QCheckBox::indicator:checked { background: #00bcd4; border-color: #00bcd4; }
    QTextEdit { background-color: #1a1a1a; color: #b0b0b0; border: 1px solid #333; border-radius: 4px; font-family: monospace; }
    QStatusBar { background-color: #1a1a1a; color: #90a0a0; }
    QScrollArea { border: none; background: transparent; }
"""

ACCENT_COLOR = "#00bcd4"
BG_COLOR = "#121212"
FONT_FAMILY = "Segoe UI"
