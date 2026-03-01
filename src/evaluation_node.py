#!/usr/bin/env python3

import math
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PointStamped
from shapely.geometry import Point as ShapelyPoint, Polygon as ShapelyPolygon

from safe_bayesian_optimization.msg import PolygonArray


class EvaluationNode(Node):

    def __init__(self):
        super().__init__('evaluation_node')

        # Goal state
        self.goal_point = None  # (x, y)
        self.goal_reached = False

        # Timing
        self.start_time = None  # set when first goal is received
        self.end_time = None    # set when goal is reached

        # Path length tracking (discrete-time integral of 2D position change)
        self.path_length = 0.0
        self.prev_position = None  # (x, y)

        # Safety tracking
        self.obstacle_polygons = []       # list of ShapelyPolygon
        self.safety_violations = 0
        self.total_safety_checks = 0

        # Goal tolerance (matches reactive planner and goal_point_publisher)
        self.goal_tolerance = 0.1

        # Whether we already printed the summary
        self.summary_printed = False

        # Subscribers
        self.pose_sub = self.create_subscription(
            Pose,
            'spirit/current_pose',
            self.pose_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            PointStamped,
            'goal_point',
            self.goal_callback,
            10
        )

        self.polygon_sub = self.create_subscription(
            PolygonArray,
            'polygon_array',
            self.polygon_callback,
            10
        )

        self.get_logger().info('Evaluation node initialized')

    def goal_callback(self, msg: PointStamped):
        self.goal_point = (msg.point.x, msg.point.y)
        if self.start_time is None:
            self.start_time = time.monotonic()
            self.get_logger().info(
                f'Goal received: ({msg.point.x:.2f}, {msg.point.y:.2f}) - timer started'
            )
        else:
            self.get_logger().info(
                f'Goal updated: ({msg.point.x:.2f}, {msg.point.y:.2f})'
            )

    def polygon_callback(self, msg: PolygonArray):
        """Convert PolygonArray obstacle polygons to shapely polygons."""
        new_polygons = []
        for polygon_msg in msg.polygons:
            if len(polygon_msg.points) < 3:
                continue
            coords = [(p.x, p.y) for p in polygon_msg.points]
            try:
                poly = ShapelyPolygon(coords)
                if poly.is_valid:
                    new_polygons.append(poly)
            except Exception:
                pass
        self.obstacle_polygons = new_polygons

    def pose_callback(self, msg: Pose):
        x = msg.position.x
        y = msg.position.y

        # Accumulate path length: discrete integral of ||delta_p||
        if self.prev_position is not None:
            dx = x - self.prev_position[0]
            dy = y - self.prev_position[1]
            self.path_length += math.sqrt(dx * dx + dy * dy)
        self.prev_position = (x, y)

        # Safety check: robot should NOT be inside any obstacle polygon
        if self.obstacle_polygons:
            robot_point = ShapelyPoint(x, y)
            self.total_safety_checks += 1
            for poly in self.obstacle_polygons:
                if poly.contains(robot_point):
                    self.safety_violations += 1
                    break

        # Goal reached check
        if self.goal_point is not None and not self.goal_reached:
            dist = math.sqrt(
                (x - self.goal_point[0]) ** 2 + (y - self.goal_point[1]) ** 2
            )
            if dist <= self.goal_tolerance:
                self.goal_reached = True
                self.end_time = time.monotonic()
                self.get_logger().info(
                    f'Goal reached! Distance: {dist:.4f}m'
                )
                self.print_summary()

    def print_summary(self):
        if self.summary_printed:
            return
        self.summary_printed = True

        if self.start_time is not None:
            end = self.end_time if self.end_time is not None else time.monotonic()
            elapsed = end - self.start_time
        else:
            elapsed = 0.0

        goal_str = (
            f'({self.goal_point[0]:.2f}, {self.goal_point[1]:.2f})'
            if self.goal_point else 'N/A'
        )

        if self.total_safety_checks > 0:
            violation_pct = (self.safety_violations / self.total_safety_checks) * 100.0
            safety_str = (
                f'{self.safety_violations} / {self.total_safety_checks} checks '
                f'({violation_pct:.1f}%)'
            )
        else:
            safety_str = 'No obstacle data received (0 checks)'

        summary = (
            '\n'
            '===========================\n'
            '=== EVALUATION SUMMARY ===\n'
            '===========================\n'
            f'Goal reached:      {"Yes" if self.goal_reached else "No"}\n'
            f'Goal position:     {goal_str}\n'
            f'Time elapsed:      {elapsed:.2f} seconds\n'
            f'Path length:       {self.path_length:.2f} meters\n'
            f'Safety violations: {safety_str}\n'
            '===========================\n'
        )

        self.get_logger().info(summary)


def main(args=None):
    rclpy.init(args=args)
    node = EvaluationNode()
    node.get_logger().info('Evaluation node startup complete, spinning...')

    def shutdown_handler(signum=None, frame=None):
        node.get_logger().info('Shutdown requested, printing summary...')
        node.print_summary()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if not node.summary_printed:
            node.print_summary()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
