import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, 
    QHBoxLayout, QProgressBar, QGroupBox, QGridLayout, QMessageBox
)
from PyQt5.QtGui import QPixmap

class GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('GUI')
        self.setGeometry(100, 100, 800, 600)  # Increased size for better display

        # Main Layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)  # Add some margins

        # Prompt Section
        prompt_layout = QHBoxLayout()
        prompt_label = QLabel('Prompt:')
        self.prompt_field = QLineEdit()
        prompt_layout.addWidget(prompt_label)
        prompt_layout.addWidget(self.prompt_field)
        main_layout.addLayout(prompt_layout)

        # Generate Image Button
        self.generate_button = QPushButton('Generate Image', self)
        self.generate_button.clicked.connect(self.generate_image)
        main_layout.addWidget(self.generate_button)

        # Image Display Section
        self.image_display_box = QGroupBox('Generated Image')
        image_display_layout = QVBoxLayout()
        self.image_display_label = QLabel()
        image_display_layout.addWidget(self.image_display_label)
        self.image_display_box.setLayout(image_display_layout)
        main_layout.addWidget(self.image_display_box)

        # Quality Progress Bars
        quality_layout = QGridLayout()  # Use grid layout for better alignment

        # LIQA Progress Bar
        liqa_label = QLabel('LIQA')
        self.liqa_progress_bar = QProgressBar(self)
        quality_layout.addWidget(liqa_label, 0, 0)
        quality_layout.addWidget(self.liqa_progress_bar, 0, 1)

        # DBCNN Progress Bar
        dbcnn_label = QLabel('DBCNN')
        self.dbcnn_progress_bar = QProgressBar(self)
        quality_layout.addWidget(dbcnn_label, 1, 0)
        quality_layout.addWidget(self.dbcnn_progress_bar, 1, 1)

        # MIAQ Progress Bar
        miaq_label = QLabel('MIAQ')
        self.miaq_progress_bar = QProgressBar(self)
        quality_layout.addWidget(miaq_label, 2, 0)
        quality_layout.addWidget(self.miaq_progress_bar, 2, 1)

        # Overall Progress Bar
        overall_label = QLabel('Overall Progress')
        self.overall_progress_bar = QProgressBar(self)
        quality_layout.addWidget(overall_label, 3, 0)
        quality_layout.addWidget(self.overall_progress_bar, 3, 1)

        # Check Quality Button
        self.quality_button = QPushButton('Check Quality', self)
        self.quality_button.clicked.connect(self.check_quality)
        quality_layout.addWidget(self.quality_button, 4, 0, 1, 2)  # Span the button across two columns

        # Align Quality Layout to the Right
        quality_layout_container = QHBoxLayout()
        quality_layout_container.addStretch()
        quality_layout_container.addLayout(quality_layout)
        main_layout.addLayout(quality_layout_container)

        # Set main layout
        self.setLayout(main_layout)

        # Add some styles
        self.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                transition-duration: 0.4s;
                cursor: pointer;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        self.show()

    def generate_image(self):
        # Placeholder for generating image logic
        self.image_display_label.setPixmap(QPixmap("main.py").scaled(600, 600, aspectRatioMode=1))
        QMessageBox.information(self, 'Generate Image', 'Generate image not implemented.')
        self.overall_progress_bar.setValue(60)

    def check_quality(self):
        # Placeholder for quality check logic
        self.liqa_progress_bar.setValue(0)
        self.dbcnn_progress_bar.setValue(0)
        self.miaq_progress_bar.setValue(0)
        QMessageBox.information(self, 'Check Quality', 'Quality check not implemented.')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = GUI()
    sys.exit(app.exec_())
