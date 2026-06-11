#!/usr/bin/env python3

import numpy as np
import rclpy
import rclpy.logging

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import QuaternionStamped
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

from transforms3d import quaternions


class AnafiLocalOdomNode(Node):

    def __init__(self):
        super().__init__('anafi_local_odom_node')
        self.get_logger().info("Initializing ANAFI local odometry node...")

        # Parameters.
        self.declare_parameter('attitude_topic', '/anafi/drone/attitude')
        self.declare_parameter('speed_topic', '/anafi/drone/speed')
        self.declare_parameter('altitude_topic', '/anafi/drone/altitude_above_to')
        self.declare_parameter('odom_topic', '/safer/localization/odom')

        self.declare_parameter('local_frame_id', 'anafi_local')
        self.declare_parameter('base_frame_id', 'anafi_base_link')

        self.declare_parameter('use_altitude_topic_for_z', True)
        self.declare_parameter('max_dt', 0.20)

        self.attitude_topic = self.get_parameter(
            'attitude_topic'
        ).get_parameter_value().string_value

        self.speed_topic = self.get_parameter(
            'speed_topic'
        ).get_parameter_value().string_value

        self.altitude_topic = self.get_parameter(
            'altitude_topic'
        ).get_parameter_value().string_value

        self.odom_topic = self.get_parameter(
            'odom_topic'
        ).get_parameter_value().string_value

        self.local_frame_id = self.get_parameter(
            'local_frame_id'
        ).get_parameter_value().string_value

        self.base_frame_id = self.get_parameter(
            'base_frame_id'
        ).get_parameter_value().string_value

        self.use_altitude_topic_for_z = self.get_parameter(
            'use_altitude_topic_for_z'
        ).get_parameter_value().bool_value

        self.max_dt = self.get_parameter(
            'max_dt'
        ).get_parameter_value().double_value

        # Local state.
        self.current_pose = np.array([0.0, 0.0, 0.0])
        self.current_rotation = np.eye(3)

        self.latest_attitude = None
        self.latest_speed = None
        self.latest_altitude = None

        self.prev_time = None

        # Subscribers.
        self.attitude_sub = self.create_subscription(
            QuaternionStamped,
            self.attitude_topic,
            self.attitude_callback,
            qos_profile_sensor_data
        )

        self.speed_sub = self.create_subscription(
            Vector3Stamped,
            self.speed_topic,
            self.speed_callback,
            qos_profile_sensor_data
        )

        self.altitude_sub = self.create_subscription(
            Float32,
            self.altitude_topic,
            self.altitude_callback,
            qos_profile_sensor_data
        )

        # Publisher.
        self.odom_pub = self.create_publisher(
            Odometry,
            self.odom_topic,
            10
        )

        # Run close to ANAFI telemetry rate.
        self.rate = 1.0 / 30.0
        self.timer = self.create_timer(self.rate, self.odom_loop)

        self.get_logger().info(f"Subscribing attitude: {self.attitude_topic}")
        self.get_logger().info(f"Subscribing speed: {self.speed_topic}")
        self.get_logger().info(f"Subscribing altitude: {self.altitude_topic}")
        self.get_logger().info(f"Publishing odometry: {self.odom_topic}")

    def attitude_callback(self, msg):
        self.latest_attitude = msg

        q = msg.quaternion

        # transforms3d expects quaternion ordering [w, x, y, z].
        self.current_rotation = quaternions.quat2mat([
            q.w,
            q.x,
            q.y,
            q.z,
        ])

    def speed_callback(self, msg):
        self.latest_speed = msg

    def altitude_callback(self, msg):
        self.latest_altitude = msg

    def odom_loop(self):
        if self.latest_attitude is None:
            self.get_logger().info("Waiting for ANAFI attitude...", throttle_duration_sec=2.0)
            return

        if self.latest_speed is None:
            self.get_logger().info("Waiting for ANAFI speed...", throttle_duration_sec=2.0)
            return

        now = self.get_clock().now()

        if self.prev_time is None:
            self.prev_time = now
            return

        dt = (now - self.prev_time).nanoseconds * 1e-9
        self.prev_time = now

        if dt <= 0.0:
            return

        if dt > self.max_dt:
            self.get_logger().warn(
                f"Large odometry dt={dt:.3f}s. Skipping integration step."
            )
            return

        v_body = np.array([
            self.latest_speed.vector.x,
            self.latest_speed.vector.y,
            self.latest_speed.vector.z,
        ])

        # Convert ANAFI body-frame velocity into local/world frame.
        v_world = self.current_rotation @ v_body

        # Dead-reckoning integration.
        self.current_pose[0] += v_world[0] * dt
        self.current_pose[1] += v_world[1] * dt

        if self.use_altitude_topic_for_z and self.latest_altitude is not None:
            self.current_pose[2] = self.latest_altitude.data
        else:
            self.current_pose[2] += v_world[2] * dt

        self.publish_odometry(now, v_body, v_world)

    def publish_odometry(self, now, v_body, v_world):
        odom_msg = Odometry()

        odom_msg.header.stamp = now.to_msg()
        odom_msg.header.frame_id = self.local_frame_id
        odom_msg.child_frame_id = self.base_frame_id

        odom_msg.pose.pose.position.x = float(self.current_pose[0])
        odom_msg.pose.pose.position.y = float(self.current_pose[1])
        odom_msg.pose.pose.position.z = float(self.current_pose[2])
        odom_msg.pose.pose.orientation = self.latest_attitude.quaternion

        # In nav_msgs/Odometry, twist is usually expressed in child_frame_id.
        # Therefore we store the measured ANAFI body-frame velocity here.
        odom_msg.twist.twist.linear.x = float(v_body[0])
        odom_msg.twist.twist.linear.y = float(v_body[1])
        odom_msg.twist.twist.linear.z = float(v_body[2])

        # Optional covariance values.
        # These are rough values because this is dead reckoning.
        odom_msg.pose.covariance[0] = 0.25
        odom_msg.pose.covariance[7] = 0.25
        odom_msg.pose.covariance[14] = 0.10

        odom_msg.twist.covariance[0] = 0.10
        odom_msg.twist.covariance[7] = 0.10
        odom_msg.twist.covariance[14] = 0.10

        self.odom_pub.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)

    node = AnafiLocalOdomNode()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        rclpy.logging.get_logger("AnafiLocalOdom").info('Done')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
