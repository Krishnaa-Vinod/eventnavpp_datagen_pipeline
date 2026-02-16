""" For Testing the voxel grid ouput from visualization_node.cpp. 
Run  python3 src/slam_rgbd_event/scripts/voxel_test_vis.py to visulize the voxel grids"""
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
import cv_bridge

class VoxelViewer(Node):
    def __init__(self):
        super().__init__('voxel_viewer')
        self.bridge = cv_bridge.CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/event_camera/voxel_grid',
            self.callback,
            10
        )

    def callback(self, msg):
        # Convert ROS Image to numpy array
        voxel = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')  # shape: (height, width, 5)
        self.get_logger().info(f"Voxel shape: {voxel.shape}, dtype: {voxel.dtype}")

        # Show each channel as a grayscale image
        for i in range(voxel.shape[2]):
            img = voxel[:, :, i]
            # Normalize for display
            img_disp = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img_disp = img_disp.astype(np.uint8)
            cv2.imshow(f'Voxel Bin {i}', img_disp)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = VoxelViewer()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()