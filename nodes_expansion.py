import numpy as np
from reservoirpy import Node

#square_node:返回由输入向量和其（各分量分别）平方得到的向量连接组成的向量
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
#square_node = CustomNode(name="square_node")
# x = np.array([1, 2, 3]).reshape(1, -1)
# y = square_node(x)
# print(y)

#solesquare_node:返回输入向量各分量分别平方后得到的向量
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
# solesquare_node = SoleSquare_node(name="solesquare_node")
# x = np.array([1, 2, 3]).reshape(1, -1)
# y = solesquare_node(x)
# print(y)

