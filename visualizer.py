from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QMainWindow, QStackedWidget, QSpinBox, QHBoxLayout, QFileDialog
import numpy as np

from dataloader import save_network, load_network

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

        # These will be updated by the model we load
        self.layers = None
        self.activations = None
        self.weights = None
        # self.activations = [
        #     [0.0 for _ in range(n)] for n in self.layers
        # ]

        # weights[layer][i][j] = weight from neuron i in layer
        # to neuron j in next layer
        # self.weights = [
        #     [
        #         [0.0 for _ in range(self.layers[l + 1])]
        #         for _ in range(self.layers[l])
        #     ]
        #     for l in range(len(self.layers) - 1)
        # ]

        self.setMinimumSize(300, 400)

    # Helper functions for updating graph
    def set_activations(self, activations):
        self.activations = activations
        self.update()

    def set_weights(self, weights):
        self.weights = weights
        self.update()

    # Helper to convert normalized value back to RGB value
    def value_to_color(self, value):
        value = max(0.0, min(1.0, value)) # Clamp to [0, 1]
        red = int(255 * value) # Convert to RGB [0, 255]
        return QColor(red, 0, 0)

    def paintEvent(self, event):
        painter = QPainter(self)
        # painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        layer_spacing = width / (len(self.layers) + 1)
        neuron_radius = 14

        # Neuron positions
        positions = []
        for layer_index, neuron_count in enumerate(self.layers):
            x = (layer_index + 1) * layer_spacing
            y_spacing = height / (neuron_count + 1)

            layer_positions = []
            for i in range(neuron_count):
                y = (i + 1) * y_spacing
                layer_positions.append(QPointF(x, y)) # More precise coordinates using floats
            positions.append(layer_positions)

        # Draw edges
        for l in range(len(self.layers) - 1):
            for i, start_pos in enumerate(positions[l]):
                for j, end_pos in enumerate(positions[l + 1]):
                    weight = self.weights[l][i][j]
                    color = self.value_to_color(abs(weight))

                    pen = QPen(color, 2)
                    painter.setPen(pen)
                    painter.drawLine(start_pos, end_pos)

        # Draw neurons
        for l, layer_positions in enumerate(positions):
            for i, pos in enumerate(layer_positions):
                activation = self.activations[l][i]
                color = self.value_to_color(activation)

                painter.setPen(QPen(QColor(0, 0, 0), 2))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(pos, neuron_radius, neuron_radius)

# Widget which shows a given data point and its label
class ImagePainter(QWidget):
    def __init__(self):
        super().__init__()

        self.setMinimumSize(200, 200)
        self.array = np.zeros((28, 28))

    def update_widget(self, array):
        # Update the array with new values or reset it
        self.array = array
        self.update() # Trigger a repaint (call paintEvent)

    def paintEvent(self, event):
        # Create a QPainter object
        painter = QPainter(self)

        rows, cols = self.array.shape

        # Calculate pixel size based on the array size, make sure they are square
        side = min(self.width(), self.height())
        pixel_size = side // max(rows, cols)
        
        # Center the square
        x_offset = (self.width() - side) // 2
        y_offset = (self.height() - side) // 2

        # Paint each pixel
        for row in range(rows):
            for col in range(cols):
                intensity = int(self.array[row, col])
                intensity = max(0, min(255, intensity))
                color = QColor(intensity, intensity, intensity)
                
                painter.fillRect(
                    x_offset + col * pixel_size,
                    y_offset + row * pixel_size,
                    pixel_size,
                    pixel_size,
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

        main_layout = QVBoxLayout(self)
        sub_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        main_layout.addLayout(sub_layout)
        sub_layout.addLayout(left_layout)
        sub_layout.addLayout(right_layout)

        # Left layout (buttons, digit picker)
        self.input_label = QLabel("Enter index (0 - 9999):")
        self.input = QSpinBox()
        self.input.setMinimum(0)
        self.input.setMaximum(9999)
        self.input.setValue(0) # Set value to 1st index for painter

        self.true_value_label = QLabel("True value: None")

        self.image_widget = ImagePainter()
        self.show_image() # Update painter

        left_layout.addWidget(self.input_label)
        left_layout.addWidget(self.input)
        left_layout.addWidget(self.true_value_label)
        left_layout.addWidget(self.image_widget)
        
        # Right layout (network visualizer)
        self.network_visualizer = NetworkVisualizer(self.state)
        right_layout.addWidget(self.network_visualizer)

        # Sub layout
        self.back_button = QPushButton("Back to Menu")
        main_layout.addWidget(self.back_button)

        # Signals
        self.input.valueChanged.connect(self.show_image)
        self.back_button.clicked.connect(lambda: stack.setCurrentIndex(0))

    # Scroll through test dataset digits
    def show_image(self):
        test_dataset = self.state.test_dataset
        index = int(self.input.text())

        self.image_widget.update_widget(test_dataset.images[index])
        self.true_value_label.setText(f"True value: {np.argmax(test_dataset.labels[index])}")

# Lets you draw a digit and have the network predict its value
class DrawWidget(QWidget):
    def __init__(self, stack, state):
        super().__init__()

        self.state = state

        layout = QVBoxLayout(self)

        title = QLabel("Draw Digit")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Buttons
        back_button = QPushButton("Back")
        back_button.clicked.connect(lambda: stack.setCurrentIndex(0))

        # Layout
        layout.addWidget(title)
        layout.addWidget(back_button)

        self.setLayout(layout)

# Loads a model
class LoadWidget(QWidget):
    def __init__(self, stack, state):
        super().__init__()

        self.state = state

        layout = QVBoxLayout(self)

        title = QLabel("Load Model")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Buttons
        browse_button = QPushButton("Browse...")
        back_button = QPushButton("Back")
        browse_button.clicked.connect(self.open_file_dialog)
        back_button.clicked.connect(lambda: stack.setCurrentIndex(0))

        # Selected file text
        self.file_label = QLabel("No file selected")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setWordWrap(True)

        # Layout
        layout.addWidget(title)
        layout.addWidget(browse_button)
        layout.addWidget(self.file_label)
        layout.addWidget(back_button)

        self.setLayout(layout)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a model (.pkl file): ")

        if file_path:
            self.file_label.setText(file_path)

class MenuWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        title = QLabel("Neural Network Visualizer")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.train_btn = QPushButton("Train")
        self.test_btn = QPushButton("Test")
        self.visualize_btn = QPushButton("Visualize")
        self.draw_btn = QPushButton("Draw Digit")
        self.load_btn = QPushButton("Load Model")
        self.exit_btn = QPushButton("Exit")

        layout.addWidget(title)
        layout.addWidget(self.train_btn)
        layout.addWidget(self.test_btn)
        layout.addWidget(self.visualize_btn)
        layout.addWidget(self.draw_btn)
        layout.addWidget(self.load_btn)
        layout.addWidget(self.exit_btn)

class MainWindow(QMainWindow):
    def __init__(self, train_dataset, test_dataset):
        super().__init__()

        self.setWindowTitle("Neural Network Visualizer")
        self.setGeometry(100, 100, 700, 700)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # Application state
        self.state = AppState(train_dataset, test_dataset)

        # Widgets, pass in state so they can access dataset
        self.menu = MenuWidget()
        self.train = TrainWidget(self.stack, self.state)
        self.test = TestWidget(self.stack, self.state)
        self.visualize = VisualizeWidget(self.stack, self.state)
        self.draw = DrawWidget(self.stack, self.state)
        self.load = LoadWidget(self.stack, self.state)

        self.stack.addWidget(self.menu)
        self.stack.addWidget(self.train)
        self.stack.addWidget(self.test)
        self.stack.addWidget(self.visualize)
        self.stack.addWidget(self.draw)
        self.stack.addWidget(self.load)

        # Button connections
        self.menu.train_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.train))
        self.menu.test_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.test))
        self.menu.visualize_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.visualize))
        self.menu.draw_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.draw))
        self.menu.load_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.load))
        self.menu.exit_btn.clicked.connect(self.close)

def start(train_dataset, test_dataset):
    app = QApplication([])
    window = MainWindow(train_dataset, test_dataset)
    window.show()
    exit(app.exec())