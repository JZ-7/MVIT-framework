import numpy as np
from reservoirpy import Node

def square_forward(node: Node, x: np.ndarray) -> np.ndarray:
    return np.c_[x, np.square(x)]
def initialize(node: Node, x: np.ndarray = None, y: np.ndarray = None):
    if x is not None:
        node.set_input_dim(x.shape[1])
        node.set_output_dim(x.shape[1]*2)
        node.set_param("const1", 1)
def initialize_fb(node: Node, feedback=None):
    if node.has_feedback:
        if feedback is not None:
            node.set_feedback_dim(feedback.shape[1])
class Square_node(Node):
    def __init__(self, name=None):
        super().__init__(
            forward=square_forward,
            initializer=initialize,
            fb_initializer=initialize_fb,
            params={"const1": None},
            name=name
        )

def solesquare_forward(node: Node, x: np.ndarray) -> np.ndarray:
    return np.square(x)
def solesquare_initialize(node: Node, x: np.ndarray = None, y: np.ndarray = None):
    if x is not None:
        node.set_input_dim(x.shape[1])
        node.set_output_dim(x.shape[1]*2)
class SoleSquare_node(Node):
    def __init__(self, name=None):
        super().__init__(
            forward=solesquare_forward,
            initializer=solesquare_initialize,
            name=name
        )

