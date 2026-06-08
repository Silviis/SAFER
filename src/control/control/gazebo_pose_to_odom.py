#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, TransformStamped
from tf2_ros import TransformBroadcaster


class GazeboPoseToOdom(Node):

    def __init__(self):
        super().__init__('gazebo_pose_to_odom')

        # params
        self.declare_parameter('model_name', 'anafi4k')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('child_frame', 'anafi4k/base_link')
        self.declare_parameter('input_topic', '/gz/pose_info')
        self.declare_parameter('output_topic', '/anafi/odometry')

        self.model_name = self.get_parameter('model_name').value
        self.world_frame = self.get_parameter('world_frame').value
        self.child_frame = self.get_parameter('child_frame').value

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.odom_pub = self.create_publisher(Odometry, output_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(
            PoseArray,
            input_topic,
            self.callback,
            10
        )

        self.get_logger().info("gazebo_pose_to_odom started")

    def callback(self, msg):

        if not msg.poses:
            return

        # -----------------------------
        # SAFE selection (temporary fix)
        # -----------------------------
        # TODO: replace with proper model lookup if PoseArray supports names
        pose = msg.poses[2]

        now = self.get_clock().now().to_msg()

        # -----------------------------
        # Odometry
        # -----------------------------
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = self.world_frame
        odom.child_frame_id = self.child_frame

        odom.pose.pose = pose

        self.odom_pub.publish(odom)

        # -----------------------------
        # TF: world → base_link
        # -----------------------------
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self.world_frame
        t.child_frame_id = self.child_frame

        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z
        t.transform.rotation = pose.orientation

        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = GazeboPoseToOdom()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()