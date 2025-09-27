import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
import numpy as np
from gripper_arm_control.utils import rmse

class SlipDetectionNode(Node):
    def __init__(self):
        super().__init__('slip_detection_node')
        self.create_subscription(Float32MultiArray, 'gripper/force', self.force_cb, 10)
        self.slip_pub = self.create_publisher(Bool, 'gripper/slip', 10)
        self.buffer = []
        self.window = 16
        self.threshold_micro = 0.90
        self.threshold_macro = 0.80

    def force_cb(self, msg):
        forces = np.array(msg.data)
        norm = np.linalg.norm(forces)
        self.buffer.append(norm)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
        prob = self.estimate_slip_probability(np.array(self.buffer))
        slip_msg = Bool()
        slip_msg.data = (prob >= self.threshold_micro)
        self.slip_pub.publish(slip_msg)

    def estimate_slip_probability(self, window):
        if window.size == 0:
            return 0.0
        mean = np.mean(window)
        std = np.std(window)
        # heuristic: low force + high std -> slip
        prob = 0.5 + 0.5 * (std / (mean + 1e-6))
        prob = float(np.clip(prob, 0.0, 1.0))
        return prob

def main(args=None):
    rclpy.init(args=args)
    node = SlipDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
