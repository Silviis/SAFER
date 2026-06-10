#!/usr/bin/env python3

import math
from enum import Enum, auto

import numpy as np
import signal


import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_srvs.srv import Trigger
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from anafi_autonomy.msg import VelocityCommand


class State(Enum):
    INIT = auto()
    TAKEOFF = auto()
    CLIMB = auto()
    TRANSIT = auto()
    SEARCH = auto()
    RETURN_HOME = auto()
    LAND = auto()
    DONE = auto()


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class ErgodicInspectionNode(Node):

    def __init__(self):
        super().__init__("ergodic_inspection_node")

        self.declare_parameter("odom_topic", "/safer/localization/odom")
        self.declare_parameter("velocity_topic", "/anafi/drone/reference/velocity")
        self.declare_parameter("takeoff_service", "/anafi/drone/takeoff")
        self.declare_parameter("land_service", "/anafi/drone/land")
        self.declare_parameter("velocity_command_frame", "body")

        self.declare_parameter("takeoff_altitude", 1.5)
        self.declare_parameter("inspection_distance", 0.48)
        self.declare_parameter("search_duration", 20.0)

        self.declare_parameter("area_width", 2.0)
        self.declare_parameter("area_depth", 1.0)
        self.declare_parameter("ellipse_area", True)

        self.declare_parameter("max_velocity", 0.5)
        self.declare_parameter("max_vertical_velocity", 0.35)
        self.declare_parameter("max_yaw_rate_deg", 60.0)

        self.declare_parameter("kp_altitude", 0.8)
        self.declare_parameter("kp_transit", 0.6)
        self.declare_parameter("kp_return", 0.6)
        self.declare_parameter("kp_yaw", 1.8)
        self.declare_parameter("kp_boundary", 0.8)
        self.declare_parameter("lock_start_yaw", True)

        self.declare_parameter("home_tolerance", 0.15)
        self.declare_parameter("center_tolerance", 0.35)
        self.declare_parameter("altitude_tolerance", 0.15)

        self.declare_parameter("fourier_order", 6)
        self.declare_parameter("ergodic_lambda_s", 1.5)
        self.declare_parameter("sigma_x", 0.35)
        self.declare_parameter("sigma_y", 0.20)

        self.odom_topic = self.get_parameter("odom_topic").value
        self.velocity_topic = self.get_parameter("velocity_topic").value
        self.takeoff_service = self.get_parameter("takeoff_service").value
        self.land_service = self.get_parameter("land_service").value
        self.velocity_command_frame = self.get_parameter("velocity_command_frame").value

        self.takeoff_altitude = float(self.get_parameter("takeoff_altitude").value)
        self.inspection_distance = float(self.get_parameter("inspection_distance").value)
        self.search_duration = float(self.get_parameter("search_duration").value)

        self.area_width = float(self.get_parameter("area_width").value)
        self.area_depth = float(self.get_parameter("area_depth").value)
        self.ellipse_area = bool(self.get_parameter("ellipse_area").value)

        self.max_velocity = float(self.get_parameter("max_velocity").value)
        self.max_vertical_velocity = float(self.get_parameter("max_vertical_velocity").value)
        self.max_yaw_rate_deg = float(self.get_parameter("max_yaw_rate_deg").value)

        self.kp_altitude = float(self.get_parameter("kp_altitude").value)
        self.kp_transit = float(self.get_parameter("kp_transit").value)
        self.kp_return = float(self.get_parameter("kp_return").value)
        self.kp_yaw = float(self.get_parameter("kp_yaw").value)
        self.kp_boundary = float(self.get_parameter("kp_boundary").value)
        self.lock_start_yaw = bool(self.get_parameter("lock_start_yaw").value)

        self.home_tolerance = float(self.get_parameter("home_tolerance").value)
        self.center_tolerance = float(self.get_parameter("center_tolerance").value)
        self.altitude_tolerance = float(self.get_parameter("altitude_tolerance").value)

        self.K = int(self.get_parameter("fourier_order").value)
        self.lambda_s = float(self.get_parameter("ergodic_lambda_s").value)
        self.sigma_x = float(self.get_parameter("sigma_x").value)
        self.sigma_y = float(self.get_parameter("sigma_y").value)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            qos,
        )

        self.vel_pub = self.create_publisher(
            VelocityCommand,
            self.velocity_topic,
            qos,
        )

        self.marker_pub = self.create_publisher(
            Marker,
            "/ergodic_inspection/markers",
            10,
        )

        self.takeoff_client = self.create_client(Trigger, self.takeoff_service)
        self.land_client = self.create_client(Trigger, self.land_service)

        self.state = State.INIT

        self.have_odom = False
        self.takeoff_sent = False
        self.land_sent = False

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.yaw = 0.0

        self.home_x = None
        self.home_y = None
        self.home_z = None
        self.home_yaw = None

        self.center_x = None
        self.center_y = None

        self.search_start_time = None
        self.last_search_time = None

        self.modes = []
        self.phi_k = {}
        self.ck_sum = {}
        self._prepare_fourier_terms()

        self.trajectory_points = []

        self.timer = self.create_timer(0.1, self.run_fsm_timer)
        signal.signal(signal.SIGINT, self.signal_handler)
        

        self.get_logger().info("Ergodic inspection node started.")

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z
        self.yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.have_odom = True

    def run_fsm_timer(self):
        if not self.have_odom:
            self.get_logger().warn("Waiting for odometry...", throttle_duration_sec=2.0)
            return

        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate_deg = 0.0

        if self.state == State.INIT:
            self.home_x = self.x
            self.home_y = self.y
            self.home_z = self.z
            self.home_yaw = self.yaw

            self.center_x = self.home_x + self.inspection_distance * math.cos(self.home_yaw)
            self.center_y = self.home_y + self.inspection_distance * math.sin(self.home_yaw)

            self.get_logger().info(
                f"Home set to ({self.home_x:.2f}, {self.home_y:.2f}, {self.home_z:.2f}). "
                f"Inspection center: ({self.center_x:.2f}, {self.center_y:.2f}). "
                f"Locked yaw: {math.degrees(self.home_yaw):.1f} deg"
            )

            self.state = State.TAKEOFF

        elif self.state == State.TAKEOFF:
            self.send_takeoff()
            self.state = State.CLIMB

        elif self.state == State.CLIMB:
            vz = self.altitude_controller(self.takeoff_altitude)
            yaw_rate_deg = self.yaw_controller()

            if abs(self.takeoff_altitude - self.z) < self.altitude_tolerance:
                self.get_logger().info("Reached inspection altitude. Transiting to inspection area.")
                self.state = State.TRANSIT

        elif self.state == State.TRANSIT:
            vx, vy = self.go_to_xy_controller(
                self.center_x,
                self.center_y,
                self.kp_transit,
                self.max_velocity,
            )
            vz = self.altitude_controller(self.takeoff_altitude)
            yaw_rate_deg = self.yaw_controller()

            dist = math.hypot(self.center_x - self.x, self.center_y - self.y)
            if dist < self.center_tolerance:
                self.get_logger().info("Reached inspection area. Starting ergodic search.")
                self.reset_ergodic_memory()
                self.search_start_time = self.now_seconds()
                self.last_search_time = self.search_start_time
                self.state = State.SEARCH

        elif self.state == State.SEARCH:
            vx, vy = self.ergodic_controller()
            vz = self.altitude_controller(self.takeoff_altitude)
            yaw_rate_deg = self.yaw_controller()

            elapsed = self.now_seconds() - self.search_start_time
            if elapsed > self.search_duration:
                self.get_logger().info("Search duration complete. Returning home.")
                self.state = State.RETURN_HOME

        elif self.state == State.RETURN_HOME:
            vx, vy = self.return_home_controller()
            vz = self.altitude_controller(self.takeoff_altitude)
            yaw_rate_deg = self.yaw_controller()

            dist = math.hypot(self.home_x - self.x, self.home_y - self.y)
            if dist < self.home_tolerance:
                self.get_logger().info("Home reached. Landing.")
                self.state = State.LAND

        elif self.state == State.LAND:
            self.publish_velocity(0.0, 0.0, 0.0, 0.0)
            self.send_land()
            self.state = State.DONE

        elif self.state == State.DONE:
            self.publish_velocity(0.0, 0.0, 0.0, 0.0)
            self.publish_markers()
            return

        vx, vy = self.saturate_xy(vx, vy, self.max_velocity)
        vz = clamp(vz, -self.max_vertical_velocity, self.max_vertical_velocity)

        self.publish_velocity(vx, vy, vz, yaw_rate_deg)
        self.publish_markers()

    def altitude_controller(self, target_z):
        return self.kp_altitude * (target_z - self.z)

    def go_to_xy_controller(self, target_x, target_y, kp, max_vel):
        ex = target_x - self.x
        ey = target_y - self.y
        vx = kp * ex
        vy = kp * ey
        return self.saturate_xy(vx, vy, max_vel)

    def return_home_controller(self):
        return self.go_to_xy_controller(
            self.home_x,
            self.home_y,
            self.kp_return,
            self.max_velocity,
        )

    def yaw_controller(self):
        if self.lock_start_yaw and self.home_yaw is not None:
            return self.yaw_hold_controller(self.home_yaw)

        return self.yaw_tracking_controller()

    def yaw_hold_controller(self, desired_yaw):
        yaw_error = wrap_angle(desired_yaw - self.yaw)
        yaw_rate_rad = self.kp_yaw * yaw_error
        yaw_rate_deg = math.degrees(yaw_rate_rad)

        return clamp(yaw_rate_deg, -self.max_yaw_rate_deg, self.max_yaw_rate_deg)

    def yaw_tracking_controller(self):
        if self.center_x is None or self.center_y is None:
            return 0.0

        desired_yaw = math.atan2(self.center_y - self.y, self.center_x - self.x)
        return self.yaw_hold_controller(desired_yaw)

    def ergodic_controller(self):
        now = self.now_seconds()
        dt = max(1e-3, now - self.last_search_time)
        self.last_search_time = now

        elapsed = max(1e-3, now - self.search_start_time)

        qx, qy = self.global_to_domain(self.x, self.y)

        for k in self.modes:
            fk = self.fourier_basis(k, qx, qy)
            self.ck_sum[k] += fk * dt

        bx = 0.0
        by = 0.0

        for k in self.modes:
            ck = self.ck_sum[k] / elapsed
            diff = ck - self.phi_k[k]
            lam = self.lambda_k(k)

            dfx, dfy = self.fourier_gradient(k, qx, qy)

            bx += lam * diff * dfx
            by += lam * diff * dfy

        norm_b = math.hypot(bx, by)

        if norm_b < 1e-6:
            vx_local = 0.0
            vy_local = 0.0
        else:
            vx_local = -self.max_velocity * bx / norm_b
            vy_local = -self.max_velocity * by / norm_b

        vx_local, vy_local = self.apply_boundary_safety(qx, qy, vx_local, vy_local)

        if self.ellipse_area:
            vx_local, vy_local = self.apply_ellipse_safety(qx, qy, vx_local, vy_local)

        vx_global, vy_global = self.local_vector_to_global(vx_local, vy_local)

        self.trajectory_points.append(Point(x=float(self.x), y=float(self.y), z=float(self.z)))
        if len(self.trajectory_points) > 1000:
            self.trajectory_points.pop(0)

        return vx_global, vy_global

    def _prepare_fourier_terms(self):
        self.modes.clear()
        self.phi_k.clear()
        self.ck_sum.clear()

        for k1 in range(self.K + 1):
            for k2 in range(self.K + 1):
                k = (k1, k2)
                self.modes.append(k)
                self.ck_sum[k] = 0.0

        self.compute_target_fourier_coefficients()

    def compute_target_fourier_coefficients(self):
        nx = 80
        ny = 40

        xs = np.linspace(0.0, self.area_width, nx)
        ys = np.linspace(0.0, self.area_depth, ny)

        dx = self.area_width / max(1, nx - 1)
        dy = self.area_depth / max(1, ny - 1)
        dA = dx * dy

        pdf = np.zeros((nx, ny), dtype=np.float64)

        cx = self.area_width * 0.5
        cy = self.area_depth * 0.5

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                lx = x - cx
                ly = y - cy

                if self.ellipse_area:
                    a = self.area_width * 0.5
                    b = self.area_depth * 0.5
                    inside = (lx / a) ** 2 + (ly / b) ** 2 <= 1.0
                    if not inside:
                        pdf[i, j] = 0.0
                        continue

                pdf[i, j] = math.exp(
                    -0.5 * ((lx / self.sigma_x) ** 2 + (ly / self.sigma_y) ** 2)
                )

        total = float(np.sum(pdf) * dA)
        if total < 1e-12:
            raise RuntimeError("Target PDF normalization failed.")

        pdf /= total

        for k in self.modes:
            val = 0.0
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    val += self.fourier_basis(k, x, y) * pdf[i, j] * dA
            self.phi_k[k] = val

    def reset_ergodic_memory(self):
        for k in self.modes:
            self.ck_sum[k] = 0.0
        self.trajectory_points.clear()

    def global_to_domain(self, gx, gy):
        lx = gx - self.center_x
        ly = gy - self.center_y

        qx = lx + self.area_width * 0.5
        qy = ly + self.area_depth * 0.5

        qx = clamp(qx, 0.0, self.area_width)
        qy = clamp(qy, 0.0, self.area_depth)

        return qx, qy

    def local_vector_to_global(self, vx_local, vy_local):
        return vx_local, vy_local

    def fourier_basis(self, k, x, y):
        k1, k2 = k
        h = self.h_k(k)

        return (
            math.cos(k1 * math.pi * x / self.area_width)
            * math.cos(k2 * math.pi * y / self.area_depth)
            / h
        )

    def fourier_gradient(self, k, x, y):
        k1, k2 = k
        h = self.h_k(k)

        if k1 == 0:
            dfdx = 0.0
        else:
            dfdx = (
                -(k1 * math.pi / self.area_width)
                * math.sin(k1 * math.pi * x / self.area_width)
                * math.cos(k2 * math.pi * y / self.area_depth)
                / h
            )

        if k2 == 0:
            dfdy = 0.0
        else:
            dfdy = (
                -(k2 * math.pi / self.area_depth)
                * math.cos(k1 * math.pi * x / self.area_width)
                * math.sin(k2 * math.pi * y / self.area_depth)
                / h
            )

        return dfdx, dfdy

    def h_k(self, k):
        k1, k2 = k

        hx = self.area_width if k1 == 0 else self.area_width / 2.0
        hy = self.area_depth if k2 == 0 else self.area_depth / 2.0

        return math.sqrt(hx * hy)

    def lambda_k(self, k):
        k1, k2 = k
        norm_sq = k1 * k1 + k2 * k2
        return 1.0 / ((1.0 + norm_sq) ** self.lambda_s)

    def apply_boundary_safety(self, qx, qy, vx, vy):
        margin_x = 0.20 * self.area_width
        margin_y = 0.20 * self.area_depth

        if qx < margin_x:
            vx += self.kp_boundary * (margin_x - qx)

        if qx > self.area_width - margin_x:
            vx -= self.kp_boundary * (qx - (self.area_width - margin_x))

        if qy < margin_y:
            vy += self.kp_boundary * (margin_y - qy)

        if qy > self.area_depth - margin_y:
            vy -= self.kp_boundary * (qy - (self.area_depth - margin_y))

        return self.saturate_xy(vx, vy, self.max_velocity)

    def apply_ellipse_safety(self, qx, qy, vx, vy):
        lx = qx - self.area_width * 0.5
        ly = qy - self.area_depth * 0.5

        a = self.area_width * 0.5
        b = self.area_depth * 0.5

        value = (lx / a) ** 2 + (ly / b) ** 2

        if value > 0.80:
            grad_x = 2.0 * lx / (a * a)
            grad_y = 2.0 * ly / (b * b)
            grad_norm = math.hypot(grad_x, grad_y)

            if grad_norm > 1e-6:
                push = self.kp_boundary * (value - 0.80)
                vx -= push * grad_x / grad_norm
                vy -= push * grad_y / grad_norm

        return self.saturate_xy(vx, vy, self.max_velocity)

    def saturate_xy(self, vx, vy, max_vel):
        norm = math.hypot(vx, vy)
        if norm > max_vel and norm > 1e-9:
            scale = max_vel / norm
            return vx * scale, vy * scale
        return vx, vy

    def publish_velocity(self, vx, vy, vz, yaw_rate_deg):
        msg = VelocityCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.velocity_command_frame

        if self.velocity_command_frame == "body":
            vx, vy = self.local_velocity_to_body(vx, vy)

        msg.vx = float(vx)
        msg.vy = float(vy)
        msg.vz = float(vz)
        msg.yaw_rate = float(yaw_rate_deg)

        self.vel_pub.publish(msg)

    def local_velocity_to_body(self, vx, vy):
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)

        body_vx = cos_yaw * vx + sin_yaw * vy
        body_vy = -sin_yaw * vx + cos_yaw * vy

        return body_vx, body_vy

    def send_takeoff(self):
        if self.takeoff_sent:
            return

        if not self.takeoff_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn("Takeoff service not available yet.")
            return

        self.takeoff_client.call_async(Trigger.Request())
        self.takeoff_sent = True
        self.get_logger().info("Takeoff command sent.")

    def send_land(self):
        if self.land_sent:
            return

        if not self.land_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn("Land service not available yet.")
            return

        self.land_client.call_async(Trigger.Request())
        self.land_sent = True
        self.get_logger().info("Land command sent.")
    
    # ---------------- SHUTDOWN ----------------
    def signal_handler(self, sig, frame): 
            print('You pressed Ctrl+C. Turning off the controller.')
            # Stop all robots at the end
            self.land_client.call_async(Trigger.Request())
            self.stop = True

            exit()  # Force Exit

    def publish_zero(self):
        cmd = VelocityCommand()
        cmd.vx = cmd.vy = cmd.vz = 0.0
        cmd.yaw_rate = 0.0
        self.vel_pub.publish(cmd)


    def publish_markers(self):
        if self.center_x is None or self.center_y is None:
            return

        stamp = self.get_clock().now().to_msg()

        area = Marker()
        area.header.frame_id = "map"
        area.header.stamp = stamp
        area.ns = "ergodic_inspection"
        area.id = 0
        area.type = Marker.LINE_STRIP
        area.action = Marker.ADD
        area.scale.x = 0.04
        area.color.g = 1.0
        area.color.a = 1.0

        xmin = self.center_x - self.area_width * 0.5
        xmax = self.center_x + self.area_width * 0.5
        ymin = self.center_y - self.area_depth * 0.5
        ymax = self.center_y + self.area_depth * 0.5
        z = self.takeoff_altitude

        if not self.ellipse_area:
            corners = [
                (xmin, ymin),
                (xmax, ymin),
                (xmax, ymax),
                (xmin, ymax),
                (xmin, ymin),
            ]
            for x, y in corners:
                area.points.append(Point(x=float(x), y=float(y), z=float(z)))
        else:
            for i in range(73):
                th = 2.0 * math.pi * i / 72.0
                x = self.center_x + (self.area_width * 0.5) * math.cos(th)
                y = self.center_y + (self.area_depth * 0.5) * math.sin(th)
                area.points.append(Point(x=float(x), y=float(y), z=float(z)))

        center = Marker()
        center.header.frame_id = "map"
        center.header.stamp = stamp
        center.ns = "ergodic_inspection"
        center.id = 1
        center.type = Marker.SPHERE
        center.action = Marker.ADD
        center.pose.position.x = float(self.center_x)
        center.pose.position.y = float(self.center_y)
        center.pose.position.z = float(self.takeoff_altitude)
        center.scale.x = 0.25
        center.scale.y = 0.25
        center.scale.z = 0.25
        center.color.r = 1.0
        center.color.a = 1.0

        traj = Marker()
        traj.header.frame_id = "map"
        traj.header.stamp = stamp
        traj.ns = "ergodic_inspection"
        traj.id = 2
        traj.type = Marker.LINE_STRIP
        traj.action = Marker.ADD
        traj.scale.x = 0.03
        traj.color.b = 1.0
        traj.color.a = 1.0
        traj.points = list(self.trajectory_points)

        self.marker_pub.publish(area)
        self.marker_pub.publish(center)
        self.marker_pub.publish(traj)

    def now_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = ErgodicInspectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_zero()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
