import rclpy
from rclpy.node import Node
import tf_transformations
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from qpsolvers import Problem, solve_problem


class CBFBuildingController(Node):

    def __init__(self):
        super().__init__('cbf_building_controller')

        """
        Waypoints defined here that the drone will cycle through. Adjust as needed for the scenario.
        YAW angle is handled by dedicated P-controller to always face the building center.
        Building is defined as a circle with center at (0,0) and radius 3.5m.
        CBF ensures the drone stays outside a safe distance from the building so there is no need to specify corner waypoints.
        """

        ## (X, Y, Z)
        self.waypoints = [
            np.array([4.5, 4.5, 1.6]),
            np.array([4.5, -4.5, 1.6]),
            np.array([-4.5, -4.5, 1.6]),
            np.array([-4.5, 4.5, 1.6]),
        ]

        # ------------------------
        # BUILDING (circle)
        # ------------------------
        self.building_center = np.array([0.0, 0.0])
        self.building_radius = np.sqrt(2.5**2 + 2.5**2)
        self.d_safe = self.building_radius + 0.5

        self.wp_index = 0
        self.goal_thresh = 0.1

        # ------------------------
        # ROS
        # ------------------------
        self.sub_odom = self.create_subscription(
            Odometry,
            '/anafi/odometry',
            self.odom_callback,
            10
        )

        self.pub_cmd = self.create_publisher(
            Twist,
            '/anafi/cmd_vel',
            10
        )

        # ------------------------
        # STATE
        # ------------------------
        self.pos = np.zeros(3)
        self.yaw = 0.0

        self.gamma = 0.5
        self.h_pow = 1.0

        # control gains
        self.k = 0.8
        self.k_z = 0.5
        self.k_yaw = 0.9

        self.vmax = 0.9

        self.timer = self.create_timer(0.02, self.control_loop)

        self.get_logger().info("CBF controller started")

    # ------------------------
    def odom_callback(self, msg):

        self.pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        q = msg.pose.pose.orientation

        _, _, self.yaw = tf_transformations.euler_from_quaternion([
            q.x,
            q.y,
            q.z,
            q.w
        ])

    # ------------------------
    def control_loop(self):

        goal = self.waypoints[self.wp_index]

        if np.linalg.norm(goal - self.pos) < self.goal_thresh:
            self.wp_index = (self.wp_index + 1) % len(self.waypoints)
            goal = self.waypoints[self.wp_index]
        self.get_logger().info(f"Current goal: {goal}")

        # ------------------------
        # XY CONTROL (smooth nominal)
        # ------------------------
        p = self.pos[:2]
        g = goal[:2]

        error = g - p

        self.get_logger().info(f"Position: {p}, Goal: {g}, Error: {error}")

        u_nom = self.vmax * np.tanh(self.k * error)


        # ------------------------
        # CBF
        # ------------------------
        vec = p - self.building_center
        dist = np.linalg.norm(vec)

        h = dist**2 - self.d_safe**2

        if dist > 1e-6:
            grad_h = 2 * vec
        else:
            grad_h = np.array([1.0, 0.0])

        P = 2 * np.eye(2)
        q = -2 * u_nom

        G = grad_h.reshape(1, 2)
        h_val = self.gamma * h

        qp = Problem(P, q, G, np.array([h_val]))

        sol = solve_problem(qp, solver="daqp")

        #u_xy = sol.x if sol is not None else u_nom
        u_xy = u_nom
        u_world = u_xy

        c = np.cos(self.yaw)
        s = np.sin(self.yaw)

        u_body_x =  c * u_world[0] + s * u_world[1]
        u_body_y = -s * u_world[0] + c * u_world[1]

        # ------------------------
        # Z CONTROL (separate!)
        # IMPORTANT: no CBF on Z
        # ------------------------
        z_error = goal[2] - self.pos[2]
        u_z = np.clip(self.k_z * z_error, -0.5, 0.5)


        # ------------------------
        # YAW CONTROL (separate!)
        # ------------------------
        target_yaw = self.building_center
        dx = target_yaw[0] - self.pos[0]
        dy = target_yaw[1] - self.pos[1]
        desired_yaw = np.arctan2(dy, dx)
        yaw_error = np.arctan2(np.sin(desired_yaw - self.yaw),
                       np.cos(desired_yaw - self.yaw))
        u_yaw = np.clip(self.k_yaw * yaw_error, -1.0, 1.0)

        self.get_logger().info(f"desired_yaw: {desired_yaw}, yaw_error: {yaw_error}, u_yaw: {u_yaw}")


        #self.get_logger().info(f"u_nom: {u_nom}, u_xy: {u_xy}, u_z: {u_z}")
        #self.get_logger().info(f"yaw: {self.yaw}")
        # ------------------------
        # PUBLISH
        # ------------------------
        msg = Twist()
        msg.linear.x = float(u_body_x)
        msg.linear.y = float(u_body_y)
        msg.linear.z = float(u_z)
        msg.angular.z = float(u_yaw)

        self.pub_cmd.publish(msg)


def main():
    rclpy.init()
    node = CBFBuildingController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()