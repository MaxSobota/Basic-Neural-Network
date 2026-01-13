from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QMainWindow, QStackedWidget, QSpinBox, QHBoxLayout, QFileDialog, QMessageBox, QComboBox, QTextEdit
import numpy as np

from dataloader import save_network, load_network
from neuralnetwork import NeuralNetwork

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

        self.state = state

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
        painter.setRenderHint(QPainter.RenderHint.Antialiasing) # Finer edges/circles

        width = self.width()
        height = self.height()

        if self.state.model == None: # If no model loaded, don't draw anything
            point = QPointF(width / 2, height / 2)
            
            painter.drawText(point, "No Model Loaded")
            return
        
        self.neurons = self.state.model.neurons # Layer setup, like [784, 256, 10]
        self.weights = self.state.model.layers # List of numpy arrays for weights, each numpy array is 2D (weights per neuron, neurons)

        layer_spacing = width / (len(self.neurons) + 1)

        # Neuron positions
        positions = []
        for layer_index, neuron_count in enumerate(self.neurons):
            x = (layer_index + 1) * layer_spacing
            y_spacing = height / (neuron_count + 1)

            layer_positions = []
            for i in range(neuron_count):
                y = (i + 1) * y_spacing
                layer_positions.append(QPointF(x, y)) # More precise coordinates using floats
            positions.append(layer_positions)

        # Draw edges
        for l in range(len(self.neurons) - 1):
            for i, start_pos in enumerate(positions[l]):
                for j, end_pos in enumerate(positions[l + 1]):
                    weight = self.weights[l][i][j]
                    color = self.value_to_color(abs(weight))

                    pen = QPen(color, 2)
                    painter.setPen(pen)
                    painter.drawLine(start_pos, end_pos)

        # Draw neurons
        for l, layer_positions in enumerate(positions):
            neuron_radius = 100 // len(layer_positions)
            for i, pos in enumerate(layer_positions):

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
        self.layers = []
        self.activations = []
        self.learning_rate = 0

        self.main_layout = QVBoxLayout(self)
        self.layers_layout = QHBoxLayout()

        self.main_layout.addWidget(QLabel("Training")) # Title? Big font, underline, etc.
        self.main_layout.addWidget(QLabel("Array Editor"))

        self.main_layout.addLayout(self.layers_layout)

        buttons_layout = QHBoxLayout()
        self.add_button = QPushButton("Add Layer")
        self.remove_button = QPushButton("Remove Layer")

        buttons_layout.addWidget(self.add_button)
        buttons_layout.addWidget(self.remove_button)
        self.main_layout.addLayout(buttons_layout)

        self.add_button.clicked.connect(self.add_layer)
        self.remove_button.clicked.connect(self.remove_last_layer)

        self.lr_enter = QTextEdit()
        self.lr_enter.setText("1e-3") # default

        self.start_button = QPushButton("Start")
        self.back_button = QPushButton("Back to Menu")

        self.main_layout.addWidget(self.start_button)
        self.main_layout.addWidget(self.back_button)
        self.main_layout.addWidget(self.lr_enter)

        self.start_button.clicked.connect(self.build_network)
        self.back_button.clicked.connect(lambda: stack.setCurrentIndex(0))

    def add_layer(self, is_input=False):
        if len(self.layers) == 10:
            QMessageBox.warning(self, "Warning", "Maximum is 10 fully connected layers.")
            return
        
        sub_layout = QVBoxLayout()

        spinbox = QSpinBox()
        spinbox.setMinimum(1)
        spinbox.setMaximum(1_000) # Arbitrary, but this much is laggy anyway
        spinbox.setValue(1)

        activation = QComboBox()
        activation.addItems(["ReLU", "Softplus", "Softmax", "CE_Loss"])

        self.layers.append(spinbox)
        self.activations.append(activation)

        sub_layout.addWidget(spinbox)
        sub_layout.addWidget(activation)

        self.layers_layout.addLayout(sub_layout)

    def remove_last_layer(self):
        count = self.layers_layout.count()
        if count == 0: # No layers added
            return

        # Remove last layout item (layer)
        item = self.layers_layout.takeAt(count - 1)
        layout = item.layout()

        if layout != None: # Delete widgets inside the sublayout
            while layout.count():
                child = layout.takeAt(0)
                widget = child.widget()
                if widget != None:
                    widget.deleteLater()

            # Delete sublayout
            layout.deleteLater()

        # Remove from other lists
        self.layers.pop()
        self.activations.pop()

    def get_values(self): # List format for NeuralNetwork class
        try:
            learning_rate = float(self.lr_enter.toPlainText())
        except ValueError:
            QMessageBox.critical(self, "Invalid number", "Not a valid learning rate. Please enter a learning rate in the form '1e-3', within the range 1e-7 to 1.")
            return None, None, None
        
        layers = [spinbox.value() for spinbox in self.layers]
        activations = [item.currentText() for item in self.activations]
        
        return layers, activations, learning_rate
    
    def build_network(self): # Builds a NeuralNetwork object from the setup made by the user
        self.layers, self.activations, self.learning_rate = self.get_values()

        # Error messages
        if self.layers == []:
            QMessageBox.critical(self, "Error", "Error creating model: Layers empty")
            return

        if self.activations == []:
            QMessageBox.critical(self, "Error", "Error creating model: Activations empty")
            return
        
        if self.learning_rate == None:
            QMessageBox.critical(self, "Error", "Error creating model: Learning rate incorrect")
            return

        return NeuralNetwork(len(self.layers), self.layers, self.activations, self.learning_rate)

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
        self.filepath = None

        layout = QVBoxLayout(self)

        title = QLabel("Load Model")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Buttons
        browse_button = QPushButton("Browse...")
        load_button = QPushButton("Load Selected")
        back_button = QPushButton("Back")
        browse_button.clicked.connect(self.open_file_dialog)
        load_button.clicked.connect(self.load_model) 
        back_button.clicked.connect(lambda: stack.setCurrentIndex(0))

        # Selected file text
        self.file_label = QLabel("No file selected")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setWordWrap(True)

        # Layout
        layout.addWidget(title)
        layout.addWidget(browse_button)
        layout.addWidget(self.file_label)
        layout.addWidget(load_button)
        layout.addWidget(back_button)

        self.setLayout(layout)

    # Load model from given filepath
    def load_model(self):
        if self.filepath != None and self.filepath != "":
            self.state.model = load_network(self.filepath)

            if self.state.model != None:
                if not isinstance(self.state.model, NeuralNetwork): # Make sure .pkl is a NeuralNetwork before working with it
                    QMessageBox.critical(self, "Error", ".pkl file is not a NeuralNetwork class!")
                    self.state.model = None
                    return

                QMessageBox.information(self, "Success", "Model successfully loaded.")
        else:
            QMessageBox.warning(self, "Error", "No filepath given.")

    def open_file_dialog(self):
        self.filepath, _ = QFileDialog.getOpenFileName(self, "Select a model (.pkl file): ")

        if self.filepath != None and self.filepath != "":
            self.file_label.setText(self.filepath)
        else:
            self.file_label.setText("No file selected")

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