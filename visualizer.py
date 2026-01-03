from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QMainWindow, QStackedWidget, QSpinBox
import numpy as np

"""
Ideas:

- Show entire neural net structure
- Show image based on index (use dataloader)
- Show color intensity based on input values
- Show prediction for image (color output neurons)

- Add custom drawing feature?
"""

# Info the widgets need
class AppState:
    def __init__(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = None
        self.training = False

# Widget which shows the neural network structure and firing
class NetworkVisualizer(QWidget):
    def __init__(self, state):
        super().__init__()

    def show_network(self):
        pass

# Widget which shows a given data point and its label
class ImagePainter(QWidget):
    def __init__(self):
        super().__init__()

        self.setMinimumSize(300, 300) # Set the minimum size for the widget
        self.array = np.zeros((28, 28))

    def update_widget(self, array):
        # Update the array with new values or reset it
        self.array = array
        self.update() # Trigger a repaint (call paintEvent)

    def paintEvent(self, event):
        # Create a QPainter object
        painter = QPainter(self)

        height, width = self.array.shape

        # Calculate pixel size based on the array size
        pixel_width = self.width() // width
        pixel_height = self.height() // height

        for row in range(height):
            for col in range(width):
                intensity = int(self.array[row, col])
                intensity = max(0, min(255, intensity))
                color = QColor(intensity, intensity, intensity)
                
                painter.fillRect(
                    col * pixel_width,
                    row * pixel_height,
                    pixel_width,
                    pixel_height,
                    color
                )

        # End the painting
        painter.end()

# Network training widget
class TrainWidget(QWidget):
    def __init__(self, stack, state):
        super().__init__()

        self.state = state

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Training"))
        self.start_button = QPushButton("Start")
        self.back_button = QPushButton("Back to Menu")

        layout.addWidget(self.start_button)
        layout.addWidget(self.back_button)

        self.back_button.clicked.connect(lambda: stack.setCurrentIndex(0))

# Network testing widget
class TestWidget(QWidget):
    def __init__(self, stack, state):
        super().__init__()

        self.state = state

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Testing"))
        self.start_button = QPushButton("Start")
        self.back_button = QPushButton("Back to Menu")

        layout.addWidget(self.start_button)
        layout.addWidget(self.back_button)

        self.back_button.clicked.connect(lambda: stack.setCurrentIndex(0))

# Has multiple ways to visualize the dataset and network
# - Allows scrolling through the test dataset (or pick an index manually) to look at premade digits
# - Allows picking an image from the test dataset and making a prediction on it
# - Allows toggling network view (shows NN structure and neuron firing strength)
# - Allows drawing your own digit (using DrawWidget) and making a prediction on it
class VisualizeWidget(QWidget):
    def __init__(self, stack, state):
        super().__init__()

        self.state = state

        layout = QVBoxLayout(self)

        self.input_label = QLabel("Enter index (0 - 9999):")
        self.input = QSpinBox()
        self.input.setMinimum(0)
        self.input.setMaximum(9999)

        self.true_value_label = QLabel("True value: None")

        self.image_widget = ImagePainter()

        self.show_button = QPushButton("Show")
        self.back_button = QPushButton("Back to Menu")

        layout.addWidget(self.input_label)
        layout.addWidget(self.input)
        layout.addWidget(self.true_value_label)
        layout.addWidget(self.show_button)
        layout.addWidget(self.image_widget)
        layout.addWidget(self.back_button)

        self.show_button.clicked.connect(self.show_image)
        self.back_button.clicked.connect(lambda: stack.setCurrentIndex(0))

    # Scroll through test dataset digits
    def show_image(self):
        test_dataset = self.state.test_dataset
        index = int(self.input.text())

        self.image_widget.update_widget(test_dataset.images[index])
        self.true_value_label.setText(f"True value: {np.argmax(test_dataset.labels[index])}")

# Lets you draw a digit and have the network predict its value
class DrawWidget(QWidget):
    def __init__(self):
        super().__init__()

class MenuWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        title = QLabel("Neural Network Visualizer")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.train_btn = QPushButton("Train")
        self.test_btn = QPushButton("Test")
        self.visualize_btn = QPushButton("Visualize")
        self.exit_btn = QPushButton("Exit")

        layout.addWidget(title)
        layout.addWidget(self.train_btn)
        layout.addWidget(self.test_btn)
        layout.addWidget(self.visualize_btn)
        layout.addWidget(self.exit_btn)

class MainWindow(QMainWindow):
    def __init__(self, train_dataset, test_dataset):
        super().__init__()

        self.setWindowTitle("Neural Network Visualizer")
        self.setGeometry(300, 300, 500, 500)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Application state
        self.state = AppState(train_dataset, test_dataset)

        # Widgets, pass in state so they can access dataset
        self.menu = MenuWidget()
        self.train = TrainWidget(self.stack, self.state)
        self.test = TestWidget(self.stack, self.state)
        self.visualize = VisualizeWidget(self.stack, self.state)

        self.stack.addWidget(self.menu)
        self.stack.addWidget(self.train)
        self.stack.addWidget(self.test)
        self.stack.addWidget(self.visualize)

        # Button connections
        self.menu.train_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.train))
        self.menu.test_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.test))
        self.menu.visualize_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.visualize))
        self.menu.exit_btn.clicked.connect(self.close)

def start(train_dataset, test_dataset):
    app = QApplication([])
    window = MainWindow(train_dataset, test_dataset)
    window.show()
    exit(app.exec())