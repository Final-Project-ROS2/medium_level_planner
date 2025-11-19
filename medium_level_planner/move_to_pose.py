from std_srvs.srv import Trigger
from geometry_msgs.msg import Pose
from custom_interfaces.action import PlanComplexCartesianSteps

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

# Simulation predefined poses
SIM_HOME_POSE = Pose()
SIM_HOME_POSE.position.x = 0.12
SIM_HOME_POSE.position.y = 0.11
SIM_HOME_POSE.position.z = 1.47
SIM_HOME_POSE.orientation.x = 0.63
SIM_HOME_POSE.orientation.y = -0.62
SIM_HOME_POSE.orientation.z = 0.32
SIM_HOME_POSE.orientation.w = -0.33

SIM_READY_POSE = Pose()
SIM_READY_POSE.position.x = 0.48
SIM_READY_POSE.position.y = 0.11
SIM_READY_POSE.position.z = 1.23
SIM_READY_POSE.orientation.x = -0.71
SIM_READY_POSE.orientation.y = 0.71
SIM_READY_POSE.orientation.z = 0.00
SIM_READY_POSE.orientation.w = 0.00

SIM_ORIENT_DOWN_POSE = Pose()
SIM_ORIENT_DOWN_POSE.orientation.x = -0.69
SIM_ORIENT_DOWN_POSE.orientation.y = 0.72
SIM_ORIENT_DOWN_POSE.orientation.z = 0.00
SIM_ORIENT_DOWN_POSE.orientation.w = 0.00

# Real Hardware predefined poses
REAL_HOME_POSE = Pose()
REAL_HOME_POSE.position.x = 0.132
REAL_HOME_POSE.position.y = 0.144
REAL_HOME_POSE.position.z = 0.5
REAL_HOME_POSE.orientation.x = 1.0
REAL_HOME_POSE.orientation.y = 0.0
REAL_HOME_POSE.orientation.z = 0.0
REAL_HOME_POSE.orientation.w = -0.312

REAL_READY_POSE = Pose()
REAL_READY_POSE.position.x = 0.131
REAL_READY_POSE.position.y = 0.2982
REAL_READY_POSE.position.z = 0.303
REAL_READY_POSE.orientation.x = 1.0
REAL_READY_POSE.orientation.y = 0.0
REAL_READY_POSE.orientation.z = 0.0
REAL_READY_POSE.orientation.w = 0.00

REAL_ORIENT_DOWN_POSE = Pose()
REAL_ORIENT_DOWN_POSE.orientation.x = 1.0
REAL_ORIENT_DOWN_POSE.orientation.y = 0.0
REAL_ORIENT_DOWN_POSE.orientation.z = 0.0
REAL_ORIENT_DOWN_POSE.orientation.w = 0.00

class MoveToPoseService(Node):

    def __init__(self):
        super().__init__('move_to_pose_service')

        self.declare_parameter("real_hardware", False)
        self.real_hardware: bool = self.get_parameter("real_hardware").get_parameter_value().bool_value
        if self.real_hardware:
            self.get_logger().info("Running in REAL HARDWARE mode.")
        else:
            self.get_logger().info("Running in SIMULATION mode.")

        # Action clients (motion / robot state)
        self.move_action_client = ActionClient(self, PlanComplexCartesianSteps, "/plan_complex_cartesian_steps")

        self.move_to_home_srv = self.create_service(Trigger, 'move_to_home', self.move_to_home_callback)
        self.move_to_ready_srv = self.create_service(Trigger, 'move_to_ready', self.move_to_ready_callback)
        

    def move_to_home_callback(self, request, response):
        self.get_logger().info('Incoming request to move to home position')
        try:
            home_pose = SIM_HOME_POSE if not self.real_hardware else REAL_HOME_POSE

            goal = PlanComplexCartesianSteps.Goal()
            goal.target_pose = home_pose

            if not self.move_action_client.wait_for_server(timeout_sec=5.0):
                return "Move action server unavailable"
            send_future = self.move_action_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_future)
            goal_handle = send_future.result()
            if not goal_handle.accepted:
                return "Move action rejected"
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result().result
            response.success = True
            response.message = "Moved to home position"
            return response
        except Exception as e:
            return f"ERROR: {e}"



    def move_to_ready_callback(self, request, response):
        self.get_logger().info('Incoming request to move to ready position')
        try:
            ready_pose = SIM_READY_POSE if not self.real_hardware else REAL_READY_POSE

            goal = PlanComplexCartesianSteps.Goal()
            goal.target_pose = ready_pose

            if not self.move_action_client.wait_for_server(timeout_sec=5.0):
                return "Move action server unavailable"
            send_future = self.move_action_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_future)
            goal_handle = send_future.result()
            if not goal_handle.accepted:
                return "Move action rejected"
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result().result
            response.success = True
            response.message = "Moved to ready position"
            return response
        except Exception as e:
            return f"ERROR: {e}"


def main():
    rclpy.init()

    move_to_pose_service = MoveToPoseService()
    rclpy.spin(move_to_pose_service)

    rclpy.shutdown()


if __name__ == '__main__':
    main()