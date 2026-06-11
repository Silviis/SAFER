#!/usr/bin/env python3
import signal
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
from transforms3d import quaternions

from anafi_autonomy.msg import VelocityCommand
from anafi_ros_interfaces.msg import GimbalCommand


class AController(Node):

    # ---------------- STATES ----------------
    MISSION_FLY = 0
    MISSION_RETURN = 1
    MISSION_LAND = 2

    FLIGHT_MOVE = 0
    FLIGHT_HOVER = 1

    SCAN_IDLE = 0
    SCAN_ACTIVE = 1
    SCAN_DONE = 2

    def __init__(self):
        super().__init__('anafi_fsm_controller')

        # ---------------- PARAMETERS ----------------
        self.declare_parameter('odom_topic', '/safer/localization/odom')
        self.declare_parameter('velocity_topic', '/anafi/drone/reference/velocity')
        self.declare_parameter('takeoff_service', '/anafi/drone/takeoff')
        self.declare_parameter('land_service', '/anafi/drone/land')

        self.declare_parameter('kp_xy', 0.12)
        self.declare_parameter('kp_z', 0.10)

        self.declare_parameter('goal_tolerance_xy', 0.12)
        self.declare_parameter('goal_tolerance_z', 0.20)

        self.declare_parameter('max_vx', 0.25)
        self.declare_parameter('max_vy', 0.25)
        self.declare_parameter('max_vz', 0.20)

        self.declare_parameter('control_period', 0.1)

        # ---------------- TOPICS ----------------
        self.odom_topic = self.get_parameter('odom_topic').value
        self.velocity_topic = self.get_parameter('velocity_topic').value
        self.takeoff_service = self.get_parameter('takeoff_service').value
        self.land_service = self.get_parameter('land_service').value

        # ---------------- CONTROL ----------------
        self.kp_xy = self.get_parameter('kp_xy').value
        self.kp_z = self.get_parameter('kp_z').value

        self.tol_xy = self.get_parameter('goal_tolerance_xy').value
        self.tol_z = self.get_parameter('goal_tolerance_z').value

        self.max_vx = self.get_parameter('max_vx').value
        self.max_vy = self.get_parameter('max_vy').value
        self.max_vz = self.get_parameter('max_vz').value

        self.rate = self.get_parameter('control_period').value

        # ---------------- MISSION ----------------

        # Y forward, X left, Z up
        self.waypoints = [
            np.array([2.0, 0.0, 1.8]),
            #np.array([2.0, 0.0, 0.8]),
            #np.array([0.0, 2.0, 0.8]),
        ]
        self.wp_idx = 0

        # ---------------- SCAN ----------------
        self.scan_pitch_angles = [-5, 0, 5, 10, 15, 0]
        self.scan_idx = 0
        self.scan_interval = 1.0
        self.scan_start_time = None

        # ---------------- FSM STATES ----------------
        self.mission_state = self.MISSION_FLY
        self.flight_state = self.FLIGHT_MOVE
        self.scan_state = self.SCAN_IDLE

        # ---------------- ROBOT STATE ----------------
        self.current_pose = np.zeros(3)
        self.current_rotation = np.eye(3)
        self.pose_received = False

        self.home_pose = None
        self.has_taken_off = False
        self.takeoff_time = None

        # ---------------- ROS ----------------
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            qos_profile_sensor_data
        )

        self.cmd_pub = self.create_publisher(
            VelocityCommand,
            self.velocity_topic,
            1
        )

        self.gimbal_pub = self.create_publisher(
            GimbalCommand,
            '/anafi/gimbal/command',
            1
        )

        self.takeoff_client = self.create_client(Trigger, self.takeoff_service)
        self.land_client = self.create_client(Trigger, self.land_service)

        self.takeoff_client.wait_for_service(10.0)
        self.land_client.wait_for_service(10.0)

        signal.signal(signal.SIGINT, self.shutdown)

        self.timer = self.create_timer(self.rate, self.control_loop)

        self.get_logger().info("3-layer FSM controller ready")

    # ---------------- ODOM ----------------
    def odom_callback(self, msg):
        self.current_pose = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ])

        q = msg.pose.pose.orientation
        self.current_rotation = quaternions.quat2mat([q.w, q.x, q.y, q.z])

        self.pose_received = True

    # ---------------- UTIL ----------------
    def saturate(self, v, lim):
        return float(np.clip(v, -lim, lim))

    def publish_zero(self):
        cmd = VelocityCommand()
        cmd.vx = cmd.vy = cmd.vz = 0.0
        cmd.yaw_rate = 0.0
        self.cmd_pub.publish(cmd)

    def publish_gimbal(self, pitch):
        cmd = GimbalCommand()
        cmd.mode = 0
        cmd.frame = 1
        cmd.roll = 0.0
        cmd.pitch = float(pitch)
        cmd.yaw = 0.0
        self.gimbal_pub.publish(cmd)

    # ---------------- TAKEOFF ----------------
    def takeoff_once(self):
        if self.has_taken_off:
            return
        self.takeoff_client.call_async(Trigger.Request())
        self.has_taken_off = True
        self.takeoff_time = self.get_clock().now()

    # ---------------- CONTROL LOOP ----------------
    def control_loop(self):

        if not self.pose_received:
            return

        if not self.has_taken_off:
            self.takeoff_once()
            return

        # stabilization
        if self.takeoff_time is not None:
            dt = (self.get_clock().now() - self.takeoff_time).nanoseconds * 1e-9
            if dt < 2.0:
                return

        if self.home_pose is None:
            self.home_pose = self.current_pose.copy()
            self.get_logger().info(f"Home: {self.home_pose}")

        # =========================================================
        # LAYER 1: MISSION FSM
        # =========================================================
        if self.mission_state == self.MISSION_FLY:
            self.get_logger().info(f"Flying to waypoint {self.wp_idx}: {self.waypoints[self.wp_idx]}", throttle_duration_sec=5.0)
            target = self.waypoints[self.wp_idx]
        elif self.mission_state == self.MISSION_LAND:
            self.publish_zero()
            self.get_logger().info("Landing")
            self.land_client.call_async(Trigger.Request())
            return
        elif self.mission_state == self.MISSION_RETURN:
            self.get_logger().info("Returning home", throttle_duration_sec=5.0)
            target = self.home_pose

        # =========================================================
        # ERROR COMPUTATION
        # =========================================================
        err_world = target - self.current_pose
        err_body = self.current_rotation.T @ err_world

        dist_xy = np.linalg.norm(err_world[:2])
        dist_z = abs(err_world[2])

        # =========================================================
        # LAYER 2: FLIGHT FSM
        # =========================================================

        if self.flight_state == self.FLIGHT_MOVE:

            cmd = VelocityCommand()
            cmd.vx = self.saturate(self.kp_xy * err_body[0], self.max_vx)
            cmd.vy = self.saturate(self.kp_xy * err_body[1], self.max_vy)
            cmd.vz = self.saturate(self.kp_z * err_body[2], self.max_vz)

            self.cmd_pub.publish(cmd)

            # waypoint reached → switch to hover + scan
            if dist_xy < self.tol_xy and dist_z < self.tol_z:
                if self.mission_state == self.MISSION_RETURN:
                    self.get_logger().info("Returning home, skipping scan")
                    self.mission_state = self.MISSION_LAND
                    return

                self.get_logger().info("Waypoint reached → HOVER + SCAN")

                self.flight_state = self.FLIGHT_HOVER
                self.scan_state = self.SCAN_ACTIVE
                self.scan_start_time = self.get_clock().now()
                self.scan_idx = 0

                return
        
        # =========================================================
        # LAYER 3: SCAN FSM
        # =========================================================
        if self.flight_state == self.FLIGHT_HOVER:

            self.publish_zero()

            if self.scan_state == self.SCAN_ACTIVE:

                now = self.get_clock().now()
                dt = (now - self.scan_start_time).nanoseconds * 1e-9

                if dt > self.scan_interval:

                    self.scan_start_time = now

                    pitch = self.scan_pitch_angles[self.scan_idx]
                    self.publish_gimbal(pitch)

                    self.get_logger().info(f"Scan pitch={pitch}")

                    self.scan_idx += 1

                    if self.scan_idx >= len(self.scan_pitch_angles):

                        self.get_logger().info("Scan complete")

                        self.flight_state = self.FLIGHT_MOVE
                        self.scan_state = self.SCAN_IDLE

                        self.wp_idx += 1

                        if self.wp_idx >= len(self.waypoints):
                            self.mission_state = self.MISSION_RETURN

                        return

    # ---------------- SHUTDOWN ----------------
    def shutdown(self, sig, frame):
        self.get_logger().info("Shutdown")
        self.publish_zero()
        self.land_client.call_async(Trigger.Request())
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = AController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()