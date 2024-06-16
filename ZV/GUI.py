from operator import mod

from PySide6 import QtCore, QtWidgets, QtGui
import os
import nnef
import nnef_tools
import tvm
import NNEFConverter

def get_content(graph):
    mod, params = NNEFConverter.from_nnef(graph)
    return mod.astext()


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
        self.textbox1 = QtWidgets.QTextEdit(self)
        self.textbox_layout.addWidget(self.textbox1)    # Add the textbox to the horizontal layout

        # Create the second QTextEdit widget for the textbox
        self.textbox2 = QtWidgets.QTextEdit(self)
        self.textbox_layout.addWidget(self.textbox2)    # Add the textbox to the horizontal layout

        # Add the horizontal layout to the overall layout
        self.layout.addLayout(self.textbox_layout)

        # Create a QPushButton for the load button
        self.load_button = QtWidgets.QPushButton("Load NNEF Graph", self)
        self.load_button.clicked.connect(self.load_text_file)
        self.layout.addWidget(self.load_button)

        self.setLayout(self.layout)

        # attribs
        self.graph = None

    def load_text_file(self):
        # Open a file dialog and get the selected file name
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open NNEF Graph",
                                                             "/home/ad.adasworks.com/agoston.czobor/.Thesis",
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
        content = get_content(self.graph)
        self.textbox2.setText(content)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    gui = GUI()
    gui.show()
    app.exec()
