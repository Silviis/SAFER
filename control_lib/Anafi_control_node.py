#!/usr/bin/env python3

import signal

import numpy as np
import rclpy
import rclpy.logging
import json
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_srvs.srv import Trigger
from transforms3d import quaternions

from anafi_autonomy.msg import VelocityCommand
from vicon_receiver.msg import Position


class AController(Node):

    def __init__(self):
        super().__init__('Kp_controller')
        self.get_logger().info("Initializing controller node...")

        self.current_pose = np.array([.0, .0, .0])
        self.current_rotation = np.eye(3)

        self.pose_received = False
        self.has_taken_off = False

        self.stop = False

        self.rate = 0.1

        # Define the goal position (x, y, z) in real-world coordinates (meters) // vicon (world)

        self.goal = np.array([1.0, 1.0, 1.0])

        self.T_12 = np.array([[0.0185, -0.9991, -0.0395, 0.0194],
                       [0.9998, 0.0183, 0.0088, -0.0281],
                       [-0.0082, -0.0397, 0.9992, -0.0069],
                       [0, 0, 0, 1]])

        self.T_01 = np.eye(4)

        # Subscribers
        self.pos_sub = self.create_subscription(
            Position,
            '/vicon/anafi_1/anafi_1',
            self.pose_callback,
            qos_profile_sensor_data
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            VelocityCommand,
            '/anafi/drone/reference/velocity',
            1
        )

        self.takeoff_client = self.create_client(Trigger, '/anafi/drone/takeoff')
        self.land_client = self.create_client(Trigger, '/anafi/drone/land')

        self.get_logger().info("Waiting for takeoff & land services to be available...")
        if not self.takeoff_client.wait_for_service(10.0):
            self.get_logger().warn(f"Waited for takeoff service: /anafi/drone/takeoff, and could not reach it.")
        if not self.land_client.wait_for_service(10.0):
            self.get_logger().warn(f"Waited for land service: /anafi/drone/land, and could not reach it.")
        self.get_logger().info("Done waiting")

        signal.signal(signal.SIGINT, self.signal_handler)

        self.trajectory = [[], [], []]# x, y, z trajectory for later plotting
        self.control_input = [[], []] # vx, vy
        self.norminal_error = []

        # start control loop
        self.timer = self.create_timer(self.rate, self.control_loop)

    def pose_callback(self, msg):

        scale = 0.001

        self.T_01[0:3, 0:3] = quaternions.quat2mat([msg.w, msg.x_rot, msg.y_rot, msg.z_rot])
        self.T_01[0, 3] = msg.x_trans * scale
        self.T_01[1, 3] = msg.y_trans * scale
        self.T_01[2, 3] = msg.z_trans * scale

        T_02 = self.T_01 @ self.T_12

        self.current_pose = T_02[0:3, 3]
        self.current_rotation = T_02[0:3, 0:3]

        self.pose_received = True


    def control_loop(self):

        if not self.pose_received:
            self.get_logger().info("Waiting for pose data...")
            return

        elif not self.has_taken_off:
            self.takeoff_client.call_async(Trigger.Request())
            self.has_taken_off = True

        error_world = self.goal - self.current_pose
        error_in_drone_frame = np.transpose(self.current_rotation) @ error_world

        self.get_logger().info(f"-----------------")
        self.get_logger().info(f"Current pose: {self.current_pose} and goal: {self.goal} error: {error_world} and error in drone frame: {error_in_drone_frame}")

        self.trajectory[0].append(self.current_pose[0])
        self.trajectory[1].append(self.current_pose[1])
        self.trajectory[2].append(self.current_pose[2])

        kp = 0.1

        ex = error_in_drone_frame[0]
        ey = error_in_drone_frame[1]
        ez = error_in_drone_frame[2]

        cmd = VelocityCommand()

        norm_er_pos = np.sqrt(error_world[0]**2 + error_world[1]**2)

        self.norminal_error.append(norm_er_pos)

        if norm_er_pos < 0.1:

            cmd.vx = 0.0
            cmd.vy = 0.0
            cmd.vz = 0.0
            cmd.yaw_rate = 0.0

            self.cmd_vel_pub.publish(cmd)

            self.get_logger().info("Goal reached")
            self.get_logger().info(f"Publishing velocity command: vx={cmd.vx}, vy={cmd.vy}, vz={cmd.vz}, yaw_rate={cmd.yaw_rate}")

            self.control_input[0].append(cmd.vx)
            self.control_input[1].append(cmd.vy)

            return

        cmd.vx = float(kp * ex)
        cmd.vy = float(kp * ey)
        cmd.vz = float(kp * ez)
        cmd.yaw_rate = 0.0

        self.get_logger().info(f"Publishing velocity command: vx={cmd.vx}, vy={cmd.vy}, vz={cmd.vz}, yaw_rate={cmd.yaw_rate}")

        self.control_input[0].append(cmd.vx)
        self.control_input[1].append(cmd.vy)

        self.cmd_vel_pub.publish(cmd)


    def signal_handler(self, sig, frame):
            print('You pressed Ctrl+C. Turning off the controller.')

            # Stop all robots at the end
            self.land_client.call_async(Trigger.Request())
            self.stop = True

            exit()  # Force Exit

def main(args=None):
    rclpy.init(args=args)

    controller = AController()

    try:

        rclpy.spin(controller)

    except (KeyboardInterrupt, SystemExit):
        rclpy.logging.get_logger("Anafi").info('Done')

    # for later on plotting the trajectories of our drone
    # save in format of x,y,z,vx,vy

    with open('/home/localadmin/anafi_gtg_t02.json','w') as out:
        json.dump({'trajectory': controller.trajectory, 'control_input': controller.control_input, 'norminal_error': controller.norminal_error}, out)

    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
