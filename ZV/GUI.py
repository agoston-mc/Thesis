from operator import mod

import numpy as np
from PyQt5.QtGui import QPixmap
from PySide6 import QtCore, QtWidgets, QtGui
import os
import nnef
import _nnef
import nnef_tools.interpreter.pytorch as interpreter
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor
import NNEFConverter


def fill_variables(graph):
    # some operations (conv, box, pools etc) have variable tensors as parameters, and they need to be filled
    for operation in graph.operations:
        if operation.name == "variable":
            tensor_name = operation.outputs["output"]

            shape = operation.attribs["shape"]

            # they are only scalar in these cases, so float32 is fine for default value
            assert (
                    operation.dtype == "scalar"
            ), f"variable of type {operation.dtype} is not supported, please update fill_variables"

            data = np.ones(shape).astype("float32")

            tensor = graph.tensors[tensor_name]
            graph.tensors[tensor_name] = _nnef.Tensor(
                tensor.name, tensor.dtype, shape, data, tensor.quantization
            )


def get_nnef_outputs(path, inputs):
    ip = interpreter.Interpreter(path, None, None)
    inputs = [inputs[tensor.name] for tensor in ip.input_details()]
    return ip(inputs)


def load_image():
    import PIL.Image

    img = PIL.Image.open('./examples/inception_v1/elephant.jpg')

    im = img.resize((224, 224))
    im = im.convert('RGB')

    # im.show()

    x = np.array(im, dtype='float32').transpose(2, 0, 1)
    x = np.expand_dims(x, axis=0)

    return x

class BenchmarkWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Create a QVBoxLayout for the overall layout
        self.layout = QtWidgets.QVBoxLayout(self)

        self.bench_button = QtWidgets.QPushButton("Run benchmarks", self)
        self.bench_button.clicked.connect(self.run_benchmarks)
        self.layout.addWidget(self.bench_button)

        # Create a QHBoxLayout for the textboxes
        self.textbox_layout = QtWidgets.QHBoxLayout()

        # Create the first QTextEdit widget for the textbox
        self.vbox1 = QtWidgets.QVBoxLayout()
        self.label1 = QtWidgets.QLabel("CPU - LLVM", self)
        self.vbox1.addWidget(self.label1)
        self.textbox1 = QtWidgets.QTextEdit(self)
        self.vbox1.addWidget(self.textbox1)
        self.textbox_layout.addLayout(self.vbox1)  # Add the textbox to the horizontal layout
        self.textbox1.setDisabled(True)

        # Create the second QTextEdit widget for the textbox
        self.vbox2 = QtWidgets.QVBoxLayout()
        self.label2 = QtWidgets.QLabel("GPU - METAL", self)
        self.vbox2.addWidget(self.label2)
        self.textbox2 = QtWidgets.QTextEdit(self)
        self.vbox2.addWidget(self.textbox2)
        self.textbox_layout.addLayout(self.vbox2)  # Add the textbox to the horizontal layout
        self.textbox2.setDisabled(True)

        # Add the horizontal layout to the overall layout
        self.layout.addLayout(self.textbox_layout)

        self.new_textbox_layout = QtWidgets.QHBoxLayout()

        # Create the third QTextEdit widget for the example run data
        self.vbox3 = QtWidgets.QVBoxLayout()
        self.label3 = QtWidgets.QLabel("Example Run - CPU", self)
        self.vbox3.addWidget(self.label3)
        self.textbox3 = QtWidgets.QTextEdit(self)
        self.vbox3.addWidget(self.textbox3)
        self.new_textbox_layout.addLayout(self.vbox3)  # Add the textbox to the new layout
        self.textbox3.setDisabled(True)

        # Create the fourth QTextEdit widget for the example run data
        self.vbox4 = QtWidgets.QVBoxLayout()
        self.label4 = QtWidgets.QLabel("Example Run - GPU", self)
        self.vbox4.addWidget(self.label4)
        self.textbox4 = QtWidgets.QTextEdit(self)
        self.vbox4.addWidget(self.textbox4)
        self.new_textbox_layout.addLayout(self.vbox4)  # Add the textbox to the new layout
        self.textbox4.setDisabled(True)

        # Add the new layout to the overall layout
        self.layout.addLayout(self.new_textbox_layout)

    def run_benchmarks(self):
        for target in ['llvm', 'cuda']:
            # Run the benchmarks
            mod, tparams = NNEFConverter.from_nnef("./examples/inception_v1/inception_v1.nnef")
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target, params=tparams)
            lib.export_library("compiled_lib.so")
            lib: tvm.runtime.Module = tvm.runtime.load_module("compiled_lib.so")
            dev = tvm.cpu(0) if target == 'llvm' else tvm.metal(0)
            graphm = graph_executor.GraphModule(lib["default"](dev))
            res = graphm.benchmark(dev, "run", 10, 10)
            print(f"target: {target}, res: {res}")
            if target == 'llvm':
                self.textbox1.setText(str(res))
            else:
                self.textbox2.setText(str(res))

            # Run the example run
            x = load_image()
            #relay
            graphm.set_input("data_0", x)
            graphm.run()
            tvm_output = graphm.get_output(0).numpy()
            #nnef
            nnef_output = get_nnef_outputs("./examples/inception_v1/inception_v1.nnef", {"data_0": x})

            classes = eval(open('./examples/inception_v1/validation_utils/class_names.txt', 'r').read())

            nnef_msg = f"Maximum confidence: {max(*nnef_output['prob_1'][0])}\n"
            nnef_msg += f"Guess: {max(enumerate(nnef_output['prob_1'][0]), key=lambda y: y[1])[0]}\n"
            nnef_msg += f"Guessed class: {classes[np.argmax(nnef_output['prob_1'][0]) - 1]}\n"
            self.textbox3.setText(nnef_msg)
            tvm_msg = f"Maximum confidence: {max(*tvm_output[0])}\n"
            tvm_msg += f"Guess: {max(enumerate(tvm_output[0]), key=lambda y: y[1])[0]}\n"
            tvm_msg += f"Guessed class: {classes[np.argmax(tvm_output[0]) - 1]}\n"
            self.textbox4.setText(tvm_msg)
            # print("Class is: " + classes[np.argmax(preds) - 1])
            print('ne  conf: ', end='')
            print(max(*nnef_output['prob_1'][0]))
            print('tvm conf: ', end='')
            print(max(*tvm_output[0]))
            print('ne  guess: ', end='')
            print(max(enumerate(nnef_output['prob_1'][0]), key=lambda y: y[1])[0])
            print('tvm guess: ', end='')
            print(max(enumerate(tvm_output[0]), key=lambda y: y[1])[0])



class CustomTextEdit(QtWidgets.QTextEdit):
    focusOut = QtCore.Signal()

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.focusOut.emit()


class GUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Transformation of Neural Networks")
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setFixedSize(1000, 500)

        # Create a QVBoxLayout for the overall layout
        self.layout = QtWidgets.QVBoxLayout(self)

        # Create a QHBoxLayout for the textboxes
        self.textbox_layout = QtWidgets.QHBoxLayout()

        # Create the first QTextEdit widget for the textbox
        self.vbox1 = QtWidgets.QVBoxLayout()
        self.label1 = QtWidgets.QLabel("NNEF", self)
        self.vbox1.addWidget(self.label1)
        self.textbox1 = CustomTextEdit(self)
        self.vbox1.addWidget(self.textbox1)
        self.textbox_layout.addLayout(self.vbox1)  # Add the textbox to the horizontal layout
        self.textbox1.focusOut.connect(self.update_textbox2)

        # Create the second QTextEdit widget for the textbox
        self.vbox2 = QtWidgets.QVBoxLayout()
        self.label2 = QtWidgets.QLabel("Relay", self)
        self.vbox2.addWidget(self.label2)
        self.textbox2 = QtWidgets.QTextEdit(self)
        self.vbox2.addWidget(self.textbox2)
        self.textbox_layout.addLayout(self.vbox2)  # Add the textbox to the horizontal layout

        # Add the horizontal layout to the overall layout
        self.layout.addLayout(self.textbox_layout)

        # Create a QPushButton for the load button
        self.load_button = QtWidgets.QPushButton("Load NNEF Graph", self)
        self.load_button.clicked.connect(self.load_text_file)
        self.layout.addWidget(self.load_button)

        self.run_ones_button = QtWidgets.QPushButton("Run Relay with ones", self)
        self.run_ones_button.clicked.connect(self.run_with_ones)
        self.layout.addWidget(self.run_ones_button)

        self.params_button = QtWidgets.QPushButton("Show Parameters", self)
        self.params_button.clicked.connect(self.show_params)
        self.layout.addWidget(self.params_button)

        self.setLayout(self.layout)

        # attribs
        self.graph = None
        self.mod = None
        self.params = None
        self.dialogs = []
        self.freeze_par = False

    def load_text_file(self):
        # Open a file dialog and get the selected file name
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open NNEF Graph",
                                                             "",
                                                             "NNEF Files (*.nnef);;Directory Files (*.dir)")

        # If a file is selected
        if file_name:
            # Check if the selected file is a directory
            if os.path.isdir(file_name):
                # If it's a directory, load the graph
                self.graph = nnef.load_graph(file_name)
                with open(file_name + "/grap.nnef", 'r') as file:
                    data = file.read()
                    self.textbox1.setText(data)
            else:
                # If it's a file, load the file contents into the textbox
                with open(file_name, 'r') as file:
                    data = file.read()
                    self.textbox1.setText(data)
                    self.graph = nnef.parse_string(data)

        # Set the content of the second textbox to a variable
        self.update_textbox2()

    def update_textbox2(self):
        # If the first textbox has content
        if self.textbox1.toPlainText():
            try:
                # Parse the content of the first textbox
                self.graph = nnef.parse_string(self.textbox1.toPlainText())
                fill_variables(self.graph)
                # Convert the graph to a relay module
                self.mod, self.params = NNEFConverter.from_nnef(self.graph, self.freeze_par)
                # Set the content of the second textbox to the relay module
                self.textbox2.setText(str(self.mod))
            except Exception as e:
                # If there is an error, show the error message in the second textbox
                self.textbox2.setText(str(e))

    def run_with_ones(self):
        # Run the relay model with ones
        target = 'llvm'
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(self.mod, target=target, params=self.params)
        dev = tvm.device('cpu', 0)
        module = tvm.contrib.graph_executor.GraphModule(lib['default'](dev))
        inputs = {}
        for inp in self.graph.inputs:
            inputs[inp] = np.ones(self.graph.tensors[inp].shape, dtype='float32')
        module.set_input(**inputs)
        module.run()
        tvm_output = module.get_output(0).numpy()
        print('tvm conf: ', end='')
        # print(max(*tvm_output[0]))
        dialog = OutputDialog()
        dialog.set_output(tvm_output)
        dialog.exec()

    def show_params(self):
        for p in self.params:
            dialog = OutputDialog(p)
            dialog.set_output(self.params[p])
            dialog.show()
            # Add the dialog to the list of dialogs
            self.dialogs.append(dialog)

    def switch_freeze(self):
        # Switch the boolean value
        self.freeze_par = not self.freeze_par


class OutputDialog(QtWidgets.QDialog):
    def __init__(self, title="Output"):
        super().__init__()
        self.setWindowTitle(title)
        self.setMaximumSize(200, 200)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.output_label = QtWidgets.QLabel(self)
        self.layout.addWidget(self.output_label)
        self.setLayout(self.layout)

    def set_output(self, output):
        self.output_label.setText(str(output))

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Transformation of Neural Networks")
        self.setGeometry(100, 100, 800, 600)

        # Create the menu bar
        self.menu = self.menuBar()

        # Create the menu items
        self.file_menu = self.menu.addMenu("File")
        self.edit_menu = self.menu.addMenu("Edit")

        # Create the actions
        self.switch_to_gui = self.file_menu.addAction("Switch to Conversion")
        self.switch_to_benchmark = self.file_menu.addAction("Switch to Benchmark")

        # Create the GUI and Benchmark windows
        self.gui = GUI()
        self.benchmark = BenchmarkWindow()

        self.switch_boolean_action = self.edit_menu.addAction("Switch Freeze parameters")
        self.switch_boolean_action.triggered.connect(self.gui.switch_freeze)

        # Create a QStackedWidget and add the GUI and Benchmark windows to it
        self.stacked_widget = QtWidgets.QStackedWidget()
        self.stacked_widget.addWidget(self.gui)
        self.stacked_widget.addWidget(self.benchmark)

        # Set the central widget to the QStackedWidget
        self.setCentralWidget(self.stacked_widget)

        # Connect the actions to the methods
        self.switch_to_gui.triggered.connect(self.load_gui)
        self.switch_to_benchmark.triggered.connect(self.load_benchmark)

    def load_gui(self):
        # Switch to the GUI widget in the QStackedWidget
        self.stacked_widget.setCurrentWidget(self.gui)

    def load_benchmark(self):
        # Switch to the Benchmark widget in the QStackedWidget
        self.stacked_widget.setCurrentWidget(self.benchmark)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
