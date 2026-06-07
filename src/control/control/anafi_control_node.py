#!/usr/bin/env python3
import json
import os
import signal

import numpy as np
import rclpy
import rclpy.logging

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
from transforms3d import quaternions

from anafi_autonomy.msg import VelocityCommand


class AController(Node):

    def __init__(self):
        super().__init__('Kp_controller')
        self.get_logger().info("Initializing ANAFI odometry-based controller node...")

        # Parameters.
        self.declare_parameter('odom_topic', '/safer/localization/odom')
        self.declare_parameter('velocity_topic', '/anafi/drone/reference/velocity')
        self.declare_parameter('takeoff_service', '/anafi/drone/takeoff')
        self.declare_parameter('land_service', '/anafi/drone/land')

        self.declare_parameter('goal_x', 1.0)
        self.declare_parameter('goal_y', 1.0)
        self.declare_parameter('goal_z', 1.0)

        self.declare_parameter('kp_xy', 0.1)
        self.declare_parameter('kp_z', 0.1)

        self.declare_parameter('goal_tolerance_xy', 0.10)
        self.declare_parameter('goal_tolerance_z', 0.15)

        self.declare_parameter('max_vx', 0.25)
        self.declare_parameter('max_vy', 0.25)
        self.declare_parameter('max_vz', 0.20)
        self.declare_parameter('max_yaw_rate', 0.0)

        self.declare_parameter('auto_takeoff', True)
        self.declare_parameter('land_on_shutdown', True)

        self.declare_parameter('control_period', 0.1)
        self.declare_parameter('log_output_path', 'outputs/anafi_goto_odom.json')

        self.odom_topic = self.get_parameter(
            'odom_topic'
        ).get_parameter_value().string_value

        self.velocity_topic = self.get_parameter(
            'velocity_topic'
        ).get_parameter_value().string_value

        self.takeoff_service = self.get_parameter(
            'takeoff_service'
        ).get_parameter_value().string_value

        self.land_service = self.get_parameter(
            'land_service'
        ).get_parameter_value().string_value

        goal_x = self.get_parameter('goal_x').get_parameter_value().double_value
        goal_y = self.get_parameter('goal_y').get_parameter_value().double_value
        goal_z = self.get_parameter('goal_z').get_parameter_value().double_value

        self.goal = np.array([goal_x, goal_y, goal_z])

        self.kp_xy = self.get_parameter(
            'kp_xy'
        ).get_parameter_value().double_value

        self.kp_z = self.get_parameter(
            'kp_z'
        ).get_parameter_value().double_value

        self.goal_tolerance_xy = self.get_parameter(
            'goal_tolerance_xy'
        ).get_parameter_value().double_value

        self.goal_tolerance_z = self.get_parameter(
            'goal_tolerance_z'
        ).get_parameter_value().double_value

        self.max_vx = self.get_parameter(
            'max_vx'
        ).get_parameter_value().double_value

        self.max_vy = self.get_parameter(
            'max_vy'
        ).get_parameter_value().double_value

        self.max_vz = self.get_parameter(
            'max_vz'
        ).get_parameter_value().double_value

        self.max_yaw_rate = self.get_parameter(
            'max_yaw_rate'
        ).get_parameter_value().double_value

        self.auto_takeoff = self.get_parameter(
            'auto_takeoff'
        ).get_parameter_value().bool_value

        self.land_on_shutdown = self.get_parameter(
            'land_on_shutdown'
        ).get_parameter_value().bool_value

        self.rate = self.get_parameter(
            'control_period'
        ).get_parameter_value().double_value

        self.log_output_path = self.get_parameter(
            'log_output_path'
        ).get_parameter_value().string_value

        # State.
        self.current_pose = np.array([0.0, 0.0, 0.0])
        self.current_rotation = np.eye(3)

        self.pose_received = False
        self.has_taken_off = False
        self.goal_reached = False
        self.stop = False

        # Subscribers.
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            qos_profile_sensor_data
        )

        # Publishers.
        self.cmd_vel_pub = self.create_publisher(
            VelocityCommand,
            self.velocity_topic,
            1
        )

        # Services.
        self.takeoff_client = self.create_client(
            Trigger,
            self.takeoff_service
        )

        self.land_client = self.create_client(
            Trigger,
            self.land_service
        )

        self.get_logger().info("Waiting for takeoff & land services to be available...")

        if not self.takeoff_client.wait_for_service(10.0):
            self.get_logger().warn(
                f"Waited for takeoff service: {self.takeoff_service}, "
                "and could not reach it."
            )

        if not self.land_client.wait_for_service(10.0):
            self.get_logger().warn(
                f"Waited for land service: {self.land_service}, "
                "and could not reach it."
            )

        self.get_logger().info("Done waiting")

        signal.signal(signal.SIGINT, self.signal_handler)

        # Logs for later plotting.
        self.trajectory = [[], [], []]
        self.control_input = [[], [], []]
        self.norminal_error = []

        # Start control loop.
        self.timer = self.create_timer(self.rate, self.control_loop)

        self.get_logger().info(f"Subscribing odometry: {self.odom_topic}")
        self.get_logger().info(f"Publishing velocity: {self.velocity_topic}")
        self.get_logger().info(f"Goal position: {self.goal}")

    def odom_callback(self, msg):
        self.current_pose = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ])

        q = msg.pose.pose.orientation

        # transforms3d expects quaternion ordering [w, x, y, z].
        self.current_rotation = quaternions.quat2mat([
            q.w,
            q.x,
            q.y,
            q.z,
        ])

        self.pose_received = True

    def saturate(self, value, max_abs_value):
        return float(np.clip(value, -max_abs_value, max_abs_value))

    def publish_zero_velocity(self):
        cmd = VelocityCommand()

        cmd.vx = 0.0
        cmd.vy = 0.0
        cmd.vz = 0.0
        cmd.yaw_rate = 0.0

        self.cmd_vel_pub.publish(cmd)

        self.control_input[0].append(cmd.vx)
        self.control_input[1].append(cmd.vy)
        self.control_input[2].append(cmd.vz)

    def takeoff_once(self):
        if not self.auto_takeoff:
            return

        if self.has_taken_off:
            return

        self.takeoff_client.call_async(Trigger.Request())
        self.has_taken_off = True
        self.get_logger().info("Takeoff command sent.")

    def control_loop(self):
        if not self.pose_received:
            self.get_logger().info("Waiting for local odometry data...")
            return

        if not self.has_taken_off:
            self.takeoff_once()

        error_world = self.goal - self.current_pose
        error_in_drone_frame = np.transpose(self.current_rotation) @ error_world

        norm_error_xy = np.sqrt(
            error_world[0] ** 2
            + error_world[1] ** 2
        )

        error_z_abs = abs(error_world[2])

        self.norminal_error.append(float(norm_error_xy))

        self.trajectory[0].append(float(self.current_pose[0]))
        self.trajectory[1].append(float(self.current_pose[1]))
        self.trajectory[2].append(float(self.current_pose[2]))

        self.get_logger().info("-----------------")
        self.get_logger().info(
            f"Current pose: {self.current_pose}, "
            f"goal: {self.goal}, "
            f"error_world: {error_world}, "
            f"error_body: {error_in_drone_frame}"
        )

        if norm_error_xy < self.goal_tolerance_xy and error_z_abs < self.goal_tolerance_z:
            self.publish_zero_velocity()

            if not self.goal_reached:
                self.get_logger().info("Goal reached.")
                self.goal_reached = True

            return

        self.goal_reached = False

        ex = error_in_drone_frame[0]
        ey = error_in_drone_frame[1]
        ez = error_in_drone_frame[2]

        cmd = VelocityCommand()

        cmd.vx = self.saturate(self.kp_xy * ex, self.max_vx)
        cmd.vy = self.saturate(self.kp_xy * ey, self.max_vy)
        cmd.vz = self.saturate(self.kp_z * ez, self.max_vz)
        cmd.yaw_rate = self.saturate(0.0, self.max_yaw_rate)

        self.get_logger().info(
            f"Publishing velocity command: "
            f"vx={cmd.vx:.3f}, "
            f"vy={cmd.vy:.3f}, "
            f"vz={cmd.vz:.3f}, "
            f"yaw_rate={cmd.yaw_rate:.3f}"
        )

        self.control_input[0].append(cmd.vx)
        self.control_input[1].append(cmd.vy)
        self.control_input[2].append(cmd.vz)

        self.cmd_vel_pub.publish(cmd)

    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C. Turning off the controller.')

        self.publish_zero_velocity()

        if self.land_on_shutdown:
            self.land_client.call_async(Trigger.Request())

        self.stop = True

        self.save_logs()

        exit()

    def save_logs(self):
        output_dir = os.path.dirname(self.log_output_path)

        if output_dir != "":
            os.makedirs(output_dir, exist_ok=True)

        with open(self.log_output_path, 'w') as out:
            json.dump(
                {
                    'trajectory': self.trajectory,
                    'control_input': self.control_input,
                    'norminal_error': self.norminal_error,
                },
                out,
                indent=2,
            )

        self.get_logger().info(f"Saved log to: {self.log_output_path}")


def main(args=None):
    rclpy.init(args=args)

    controller = AController()

    try:
        rclpy.spin(controller)
    except (KeyboardInterrupt, SystemExit):
        rclpy.logging.get_logger("Anafi").info('Done')

    controller.publish_zero_velocity()
    controller.save_logs()

    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
