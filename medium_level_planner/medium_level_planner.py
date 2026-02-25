#!/usr/bin/env python3
"""
ros2_llm_agent_with_vision_tools.py

Extended Ros2 LLM Agent Node with vision tools including bbox-based services:
- /vision/detect_objects        -> custom_interfaces.srv.DetectObjects
- /vision/classify_all          -> std_srvs.srv.Trigger
- /vision/classify_bb           -> custom_interfaces.srv.ClassifyBBox
- /vision/detect_grasp          -> custom_interfaces.srv.DetectGrasps
- /vision/detect_grasp_bb       -> custom_interfaces.srv.DetectGraspBBox
- /vision/understand_scene      -> custom_interfaces.srv.UnderstandScene
- /vision/find_object           -> custom_interfaces.srv.FindObjectReal

Motion/action tools preserved:
- /plan_complex_cartesian_steps (PlanComplexCartesianSteps action)
- /plan_cartesian_relative      (MoveitRelative action)
- /get_current_pose             (GetCurrentPose action)
- /get_joint_angles             (GetJointAngles action)
- gripper control (service or action depending on real_hardware)

"""
import os
import re
import threading
import time
from typing import List, Dict, Any, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, CancelResponse, GoalResponse

# Services
from std_srvs.srv import SetBool, Trigger

# Actions
from custom_interfaces.action import Prompt
from custom_interfaces.action import GetCurrentPose
from custom_interfaces.action import GetJointAngles
from custom_interfaces.action import MoveitRelative
from custom_interfaces.action import PlanComplexCartesianSteps
from control_msgs.action import GripperCommand
from geometry_msgs.msg import Pose

# Custom vision services (assumed available)
from custom_interfaces.srv import (
    DetectObjects,
    ClassifyBBox,
    DetectGrasps,
    DetectGraspBBox,
    UnderstandScene,
    FindObjectReal,
    FindObjectGrasp,
)

from custom_interfaces.srv import GetSetBool

# LangChain / LLM - keep the same imports you used
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama


# dotenv
from dotenv import load_dotenv

ENV_PATH = '/home/group11/final_project_ws/src/medium_level_planner/.env'
load_dotenv(dotenv_path=ENV_PATH)


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

SIM_HANDOVER_POSE = Pose()
SIM_HANDOVER_POSE.position.x = 0.48
SIM_HANDOVER_POSE.position.y = -0.5
SIM_HANDOVER_POSE.position.z = 1.23
SIM_HANDOVER_POSE.orientation.x = -0.71
SIM_HANDOVER_POSE.orientation.y = 0.71
SIM_HANDOVER_POSE.orientation.z = 0.00
SIM_HANDOVER_POSE.orientation.w = 0.00

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

REAL_HANDOVER_POSE = Pose()
REAL_HANDOVER_POSE.position.x = 0.531
REAL_HANDOVER_POSE.position.y = 0.1482
REAL_HANDOVER_POSE.position.z = 0.303
REAL_HANDOVER_POSE.orientation.x = 1.0
REAL_HANDOVER_POSE.orientation.y = 0.0
REAL_HANDOVER_POSE.orientation.z = 0.0
REAL_HANDOVER_POSE.orientation.w = 0.00

REAL_ORIENT_DOWN_POSE = Pose()
REAL_ORIENT_DOWN_POSE.orientation.x = 1.0
REAL_ORIENT_DOWN_POSE.orientation.y = 0.0
REAL_ORIENT_DOWN_POSE.orientation.z = 0.0
REAL_ORIENT_DOWN_POSE.orientation.w = 0.00

# Composite tools delay time
TOOL_DELAY = 5.0

class Ros2LLMAgentNode(Node):
    def __init__(self):
        super().__init__("ros2_llm_agent")
        self.get_logger().info("Initializing Ros2 LLM Agent Node...")

        self.declare_parameter("real_hardware", False)
        self.real_hardware: bool = self.get_parameter("real_hardware").get_parameter_value().bool_value
        if self.real_hardware:
            self.get_logger().info("Running in REAL HARDWARE mode.")
        else:
            self.get_logger().info("Running in SIMULATION mode.")
        self.declare_parameter("use_ollama", False)
        self.use_ollama: bool = self.get_parameter("use_ollama").get_parameter_value().bool_value
        self.declare_parameter("ollama_model", "gpt-oss:20b")
        self.ollama_model: str = self.get_parameter("ollama_model").get_parameter_value().string_value

        # -----------------------------
        # LLM Selection: Gemini or Ollama
        # -----------------------------
        if self.use_ollama:
            self.get_logger().info("Using local LLM via Ollama.")
            # Example: using llama3.1 or any model installed in `ollama list`
            self.llm = ChatOllama(
                model=self.ollama_model,   # <--- change to any local model you want
                temperature=0.0
            )
        else:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                self.get_logger().warn("No LLM API key found in environment variables GEMINI_API_KEY.")
            self.get_logger().info("Using Google Gemini API LLM.")
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key,
                temperature=0.0,
            )

        # Action clients (motion / robot state)
        self.move_action_client = ActionClient(self, PlanComplexCartesianSteps, "/plan_complex_cartesian_steps")
        self.pose_action_client = ActionClient(self, GetCurrentPose, "/get_current_pose")
        self.joint_action_client = ActionClient(self, GetJointAngles, "/get_joint_angles")
        self.relative_action_client = ActionClient(self, MoveitRelative, "/plan_cartesian_relative")

        if self.real_hardware:
            self.gripper_client = self.create_client(SetBool, "/control_gripper")
        else:
            self.gripper_client = ActionClient(self, GripperCommand, "/gripper_wrapper")

        # Vision service clients (on-demand)
        self.vision_detect_objects_client = self.create_client(DetectObjects, "/vision/detect_objects")
        self.vision_classify_all_client = self.create_client(Trigger, "/vision/classify_all")
        self.vision_classify_bb_client = self.create_client(ClassifyBBox, "/vision/classify_bb")
        self.vision_detect_grasp_client = self.create_client(DetectGrasps, "/vision/detect_grasp")
        self.vision_detect_grasp_bb_client = self.create_client(DetectGraspBBox, "/vision/detect_grasp_bb")
        self.vision_understand_scene_client = self.create_client(UnderstandScene, "/vision/understand_scene")
        self.find_object_client = self.create_client(FindObjectReal, "/find_object")
        self.find_object_grasp_client = self.create_client(FindObjectGrasp, "/find_object_grasp")

        # PDDL state service clients
        self.is_home_client = self.create_client(GetSetBool, "/is_home")
        self.is_ready_client = self.create_client(GetSetBool, "/is_ready")
        self.gripper_is_open_client = self.create_client(GetSetBool, "/gripper_is_open")
        self.is_handover_client = self.create_client(GetSetBool, "/is_handover")

        # Shared state for tracking which tools were called during one prompt execution
        self._tools_called: List[str] = []
        self._tools_called_lock = threading.Lock()

        # Initialize LangChain tools that wrap ROS clients
        self.tools = self._initialize_tools()

        # Create the tool-calling agent (prompt template follows your example style)
        self.agent_executor = self._create_agent_executor()

        # ActionServer for Prompt.action
        self._action_server = ActionServer(
            self,
            Prompt,
            "/medium_level",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self.get_logger().info("Ros2 LLM Agent Node ready (Prompt action server running).")

    def set_robot_state(self, state_name: str, value: bool) -> bool:
        """
        Set robot state via PDDL state services.
        state_name: 'is_home', 'is_ready', 'gripper_is_open'
        value: True/False
        Returns True if successful, False otherwise.
        """
        self.get_logger().info(f"[set_robot_state] Setting {state_name} to {value}")
        client_map = {
            "is_home": self.is_home_client,
            "is_ready": self.is_ready_client,
            "gripper_is_open": self.gripper_is_open_client,
            "is_handover": self.is_handover_client,
        }
        client = client_map.get(state_name)
        if client is None:
            self.get_logger().error(f"Unknown state name: {state_name}")
            return False

        try:
            if not client.wait_for_service(timeout_sec=5.0):
                self.get_logger().error(f"Service /{state_name} unavailable")
                return False
            request = GetSetBool.Request()
            request.set = True
            request.value = value
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            response = future.result()
            if response.success:
                self.get_logger().info(f"Set {state_name} to {value}")
                return True
            else:
                self.get_logger().error(f"Failed to set {state_name}: {response.message}")
                return False
        except Exception as e:
            self.get_logger().error(f"ERROR in set_robot_state for {state_name}: {e}")
            return False
    
    # -----------------------
    # Reusable Tools
    # -----------------------

    # In your Ros2LLMAgentNode class, add these private helper methods:

    def _move_to_ready(self) -> str:
        self.get_logger().info("[_move_to_ready] Moving to ready pose")
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
        if result.success:
            self.set_robot_state("is_home", False)
            self.set_robot_state("is_ready", True)
            self.set_robot_state("is_handover", False)
        return f"move_to_ready result: success={result.success}"

    def _move_to_home(self) -> str:
        self.get_logger().info("[_move_to_home] Moving to home pose")
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
        if result.success:
            self.set_robot_state("is_home", True)
            self.set_robot_state("is_ready", False)
            self.set_robot_state("is_handover", False)
        return f"move_to_home result: success={result.success}"
    
    def _move_to_handover(self) -> str:
        self.get_logger().info("[_move_to_handover] Moving to handover pose")
        handover_pose = SIM_HANDOVER_POSE if not self.real_hardware else REAL_HANDOVER_POSE

        goal = PlanComplexCartesianSteps.Goal()
        goal.target_pose = handover_pose

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
        if result.success:
            self.set_robot_state("is_home", False)
            self.set_robot_state("is_ready", False)
            self.set_robot_state("is_handover", True)
        return f"move_to_handover result: success={result.success}"
    
    def _orient_gripper_down(self) -> str:
        self.get_logger().info("[_orient_gripper_down] Orienting gripper downward")
        goal = PlanComplexCartesianSteps.Goal()
        down_orientation = Pose()
        down_orientation.orientation.x = -0.69
        down_orientation.orientation.y = 0.72
        down_orientation.orientation.z = 0.00
        down_orientation.orientation.w = 0.00
        goal.target_pose = down_orientation

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
        return f"orient_gripper_down result: success={result.success}"

    def _set_gripper_position(self, position: float, max_effort: float) -> str:
        self.get_logger().info(f"[_set_gripper_position] position={position}, max_effort={max_effort}")
        if isinstance(self.gripper_client, ActionClient):
            goal = GripperCommand.Goal()
            goal.command.position = position
            goal.command.max_effort = max_effort
            if not self.gripper_client.wait_for_server(timeout_sec=5.0):
                return "Gripper action server unavailable"
            send_future = self.gripper_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_future)
            goal_handle = send_future.result()
            if not goal_handle.accepted:
                return "Gripper action rejected"
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            result = result_future.result().result
            if result.reached_goal:
                self.set_robot_state("gripper_is_open", position == 0.0)
            return f"set_gripper_position result: success={getattr(result, 'reached_goal', False)}"
        return "set_gripper_position not available for real hardware"

    def _close_gripper(self, close: bool) -> str:
        self.get_logger().info(f"[_close_gripper] close={close}")
        if isinstance(self.gripper_client, ActionClient):
            return "close_gripper is not available in simulation mode."
        if not self.gripper_client.wait_for_service(timeout_sec=5.0):
            return "Gripper service /control_gripper unavailable"
        request = SetBool.Request()
        request.data = close
        future = self.gripper_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response.success:
            self.set_robot_state("gripper_is_open", not close)
            return f"Gripper successfully {'closed' if close else 'opened'}."
        return f"Failed to {'close' if close else 'open'} gripper: {response.message}"

    def _find_object(self, object_name: str) -> str:
        self.get_logger().info(f"[_find_object] Searching for {object_name}")
        if not self.find_object_client.wait_for_service(timeout_sec=5.0):
            return "Service /find_object unavailable"
        req = FindObjectReal.Request()
        req.label = object_name
        future = self.find_object_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        if resp is None:
            return "No response from /find_object"
        if not resp.success:
            return f"find_object failed: {resp.error_message or 'unknown'}"
        x, y, z = resp.x, resp.y, resp.z
        if x is None or y is None or z is None:
            return f"{object_name} not found in the scene."
        if self.real_hardware:
            return f"{object_name} is at position x={x:.3f}, y={y:.3f}, z={REAL_READY_POSE.position.z:.3f}"
        else:
            return f"{object_name} is at position x={x:.3f}, y={y:.3f}, z={SIM_READY_POSE.position.z:.3f}"

    def _fetch_current_pose(self) -> Optional[Pose]:
        self.get_logger().info("[_fetch_current_pose] Requesting current pose")
        if not self.pose_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("[_fetch_current_pose] /get_current_pose unavailable")
            return None
        goal = GetCurrentPose.Goal()
        send_future = self.pose_action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("[_fetch_current_pose] Goal rejected")
            return None
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        if result.success:
            return result.pose
        self.get_logger().error("[_fetch_current_pose] Failed to fetch pose")
        return None

    def _get_current_pose(self) -> str:
        pose = self._fetch_current_pose()
        if pose is None:
            return "Failed to get current pose"
        return f"pose: {pose}"

    def _move_linear_to_pose(self, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w) -> str:
        self.get_logger().info(
            f"[_move_linear_to_pose] target=({pos_x}, {pos_y}, {pos_z}), "
            f"orientation=({rot_x}, {rot_y}, {rot_z}, {rot_w})"
        )
        goal = PlanComplexCartesianSteps.Goal()
        pose = Pose()
        pose.position.x = pos_x
        pose.position.y = pos_y
        pose.position.z = pos_z
        pose.orientation.x = rot_x
        pose.orientation.y = rot_y
        pose.orientation.z = rot_z
        pose.orientation.w = rot_w
        goal.target_pose = pose
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
        if result.success:
            self.set_robot_state("is_home", False)
            self.set_robot_state("is_ready", False)
            self.set_robot_state("is_handover", False)
        return f"move_to_pose result: success={result.success}"

    def _move_relative(self, dx, dy, dz, roll, pitch, yaw) -> str:
        self.get_logger().info(
            f"[_move_relative] dx={dx}, dy={dy}, dz={dz}, roll={roll}, pitch={pitch}, yaw={yaw}"
        )
        if not self.relative_action_client.wait_for_server(timeout_sec=5.0):
            return "Action server /plan_cartesian_relative unavailable."
        goal_msg = MoveitRelative.Goal()
        goal_msg.distance_x = dx
        goal_msg.distance_y = dy
        goal_msg.distance_z = dz
        goal_msg.roll = roll
        goal_msg.pitch = pitch
        goal_msg.yaw = yaw
        send_goal_future = self.relative_action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            return "Relative motion goal rejected by action server."
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        if result.success:
            self.set_robot_state("is_home", False)
            self.set_robot_state("is_ready", False)
            self.set_robot_state("is_handover", False)
            return (f"Relative motion executed successfully: "
                    f"Δx={dx:.3f}, Δy={dy:.3f}, Δz={dz:.3f}, "
                    f"roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")
        return "Cartesian path planning or execution failed."

    def _move_to_object(self, object_name: str) -> str:
        self.get_logger().info(f"[_move_to_object] Moving to object '{object_name}'")
        find_result = self._find_object(object_name)
        if "not found" in find_result.lower() or "failed" in find_result.lower():
            return f"move_to_object aborted: {find_result}"

        pos_match = re.search(r"x=([-0-9.eE]+).*y=([-0-9.eE]+).*z=([-0-9.eE]+)", find_result)
        if not pos_match:
            return f"Could not parse object position from: {find_result}"
        pos_x, pos_y, pos_z = map(float, pos_match.groups())

        current_pose = self._fetch_current_pose()
        if current_pose is not None:
            rot_x = current_pose.orientation.x
            rot_y = current_pose.orientation.y
            rot_z = current_pose.orientation.z
            rot_w = current_pose.orientation.w
        else:
            fallback_pose = REAL_READY_POSE if self.real_hardware else SIM_READY_POSE
            rot_x = fallback_pose.orientation.x
            rot_y = fallback_pose.orientation.y
            rot_z = fallback_pose.orientation.z
            rot_w = fallback_pose.orientation.w

        return self._move_linear_to_pose(pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w)

    # -----------------------
    # Tool wrappers (LangChain)
    # -----------------------
    def _initialize_tools(self) -> List[BaseTool]:
        """
        Define tools as @tool-decorated functions that internally call ROS service/action clients.
        Each tool appends its name to self._tools_called (thread-safe) so we can report feedback.
        """

        tools: List[BaseTool] = []

        # -------------------- Motion & State Tools --------------------

        @tool
        def get_current_pose() -> str:
            """Returns the current robot's end-effector pose relative to base_link."""
            tool_name = "get_current_pose"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)
            try:
                return self._get_current_pose()
            except Exception as e:
                return f"ERROR in {tool_name}: {e}"

        tools.append(get_current_pose)

        @tool
        def move_linear_to_pose(pos_x: float, pos_y: float, pos_z: float,
                                rot_x: float, rot_y: float, rot_z: float, rot_w: float) -> str:
            """Move robot to a target pose using quaternion orientation."""
            tool_name = "move_linear_to_pose"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)
            try:
                return self._move_linear_to_pose(pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w)
            except Exception as e:
                return f"ERROR in {tool_name}: {e}"

        tools.append(move_linear_to_pose)

        @tool
        def move_to_home() -> str:
            """
            Move robot to a predefined home pose.
            """
            tool_name = "move_to_home"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)
            try:
                return self._move_to_home()
            except Exception as e:
                return f"ERROR in {tool_name}: {e}"

        tools.append(move_to_home)

        @tool
        def move_to_ready() -> str:
            """Move robot to a predefined ready pose."""
            tool_name = "move_to_ready"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)
            try:
                return self._move_to_ready()
            except Exception as e:
                return f"ERROR in {tool_name}: {e}"

        tools.append(move_to_ready)

        @tool
        def move_to_handover() -> str:
            """
            Move robot to a predefined ready pose.
            """
            tool_name = "move_to_handover"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)

            try:
                return self._move_to_handover()
            except Exception as e:
                return f"ERROR in {tool_name}: {e}"
        
        tools.append(move_to_handover)

        @tool
        def orient_gripper_down() -> str:
            """
            Orient the gripper to face downwards.
            """
            tool_name = "orient_gripper_down"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)
            try:
                return self._orient_gripper_down()
            except Exception as e:
                return f"ERROR in {tool_name}: {e}"
        tools.append(orient_gripper_down)

        if self.real_hardware:
            @tool
            def close_gripper(close: bool) -> str:
                tool_name = "close_gripper"
                """Close or open the gripper (real hardware)."""
                with self._tools_called_lock:
                    self._tools_called.append(tool_name)
                try:
                    return self._close_gripper(close)
                except Exception as e:
                    return f"ERROR in {tool_name}: {e}"
            tools.append(close_gripper)
        else:
            @tool
            def set_gripper_position(position: float, max_effort: float) -> str:
                """Set the gripper to a specific position. 0.0=open, 0.8=closed."""
                tool_name = "set_gripper_position"
                with self._tools_called_lock:
                    self._tools_called.append(tool_name)
                try:
                    return self._set_gripper_position(position, max_effort)
                except Exception as e:
                    return f"ERROR in {tool_name}: {e}"
            tools.append(set_gripper_position)

        @tool
        def move_relative(dx: float, dy: float, dz: float,
                        roll: float, pitch: float, yaw: float) -> str:
            """Moves the robot end-effector relative to its current pose."""
            tool_name = "move_relative"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)
            try:
                return self._move_relative(dx, dy, dz, roll, pitch, yaw)
            except Exception as e:
                return f"ERROR in {tool_name}: {e}"

        tools.append(move_relative)

        @tool
        def find_object(object_name: str) -> str:
            """
            Call /find_object which returns the position of the specified object.
            """
            tool_name = "find_object"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)
            try:
                return self._find_object(object_name)
            except Exception as e:
                return f"ERROR in {tool_name}: {e}"

        tools.append(find_object)

        @tool
        def move_to_object(object_name: str) -> str:
            """
            Look up an object's position with find_object then move there preserving current orientation.
            """
            tool_name = "move_to_object"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)

            try:
                return self._move_to_object(object_name)
            except Exception as e:
                return f"ERROR in {tool_name}: {e}"

        tools.append(move_to_object)

        @tool
        def place_at(x: float, y: float, z: float) -> str:
            """
            Move to a position above the target, then opening the gripper.
            """
            tool_name = "place_at"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)

            try:
                # Move to the target
                move_to_result = self._move_linear_to_pose(x, y, z, 0, 0, 0, 1)
                if "success=False" in move_to_result:
                    return f"Failed to move to target: {move_to_result}"

                time.sleep(TOOL_DELAY)

                # Open gripper
                if self.real_hardware:
                    open_gripper_result = self._close_gripper(False)
                else:
                    open_gripper_result = self._set_gripper_position(0.0, 0.1)
                if "success=False" in open_gripper_result:
                    return f"Failed to open gripper: {open_gripper_result}"

                time.sleep(TOOL_DELAY)

                return f"Successfully placed at ({x:.3f}, {y:.3f}, {z:.3f})"

            except Exception as e:
                return f"ERROR in {tool_name}: {e}"
        
        tools.append(place_at)

        @tool
        def pickup_object(object_name: str) -> str:
            """
            Move to ready, open gripper, go to the object, descend slightly, then close the gripper.
            """
            tool_name = "pickup_object"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)

            try:
                sequence = []

                move_to_ready_result = self._move_to_ready()
                sequence.append(move_to_ready_result)

                time.sleep(TOOL_DELAY)

                if self.real_hardware:
                    open_gripper_result = self._close_gripper(False)
                else:
                    open_gripper_result = self._set_gripper_position(0.0, 0.01)
                sequence.append(open_gripper_result)

                time.sleep(TOOL_DELAY)

                move_result = self._move_to_object(object_name)
                sequence.append(move_result)
                if "success=False" in move_result or "aborted" in move_result:
                    return "; ".join(sequence)

                time.sleep(TOOL_DELAY)

                move_down_result = self._move_relative(0.0, 0.0, -0.01, 0.0, 0.0, 0.0)
                sequence.append(move_down_result)

                time.sleep(TOOL_DELAY)

                if self.real_hardware:
                    close_gripper_result = self._close_gripper(True)
                else:
                    close_gripper_result = self._set_gripper_position(0.8, 0.01)
                sequence.append(close_gripper_result)

                time.sleep(TOOL_DELAY)

                move_up_result = self._move_relative(0.0, 0.0, 0.01, 0.0, 0.0, 0.0)
                sequence.append(move_up_result)

                return f"Successfully picked up {object_name}"
            except Exception as e:
                return f"ERROR in {tool_name}: {e}"

        tools.append(pickup_object)

        return tools

    # -----------------------
    # Create agent executor
    # -----------------------
    def _create_agent_executor(self) -> AgentExecutor:
        """
        Create a LangChain tool-calling agent. The agent will call the @tool functions above as needed.
        """
        if self.real_hardware:
            system_message = (
                "You are a ROS2-capable assistant. You can call the following tools (services/actions) to "
                "query sensors, perceive the environment, or command the robot: get_current_pose, move_linear_to_pose, set_gripper_position, move_relative,"
                "move_to_home, move_to_ready, move_to_handover, orient_gripper_down, close_gripper, place_at, find_object, move_to_object, pickup_object. "
                "try to use as few tools as possible to accomplish the task.\n"
                f"Home is at {REAL_HOME_POSE}, ready is at {REAL_READY_POSE}, handover is at {REAL_HANDOVER_POSE}.\n"
                "If you are instructed to move a certain direction (e.g., UP, DOWN, FORWARD, BACKWARD, LEFT, RIGHT), use the move_relative tool with small increments (e.g., 0.05m).\n"
                "If you are instructed to move to position you don't know, make reasonable assumptions, DO NOT ask for clarification.\n"
                "When you choose to use a tool, call it with appropriate arguments (if any). "
                "Return a final, concise, actionable response after using tools.\n"
                "Here is environment guidance:\n"
                "- You can get your current gripper cartesian position by using the get_current_pose tool.\n"
                "- The direction up is along positive Z axis, down is along negative Z axis.\n"
                "- The direction forward is along negative Y axis, backward is along positive Y axis.\n"
                "- The direction left is along positive X axis, right is along negative X axis.\n"
            )
        else:
            system_message = (
                "You are a ROS2-capable assistant. You can call the following tools (services/actions) to "
                "query sensors, perceive the environment, or command the robot: get_current_pose, move_linear_to_pose, set_gripper_position, move_relative,"
                "move_to_home, move_to_ready, move_to_handover, orient_gripper_down, close_gripper, place_at, find_object, move_to_object, pickup_object. "
                "try to use as few tools as possible to accomplish the task.\n"
                f"Home is at {SIM_HOME_POSE}, ready is at {SIM_READY_POSE}, handover is at {SIM_HANDOVER_POSE}.\n"
                "If you are **EXPLICITLY** instructed to **OPEN** the gripper, set the gripper position to 0.0. "
                "If you are **EXPLICITLY** instructed to **GRAB** an object, set the gripper position to 0.2. "
                "If you are **EXPLICITLY** instructed to **CLOSE** the gripper, set the gripper position to 0.8. "
                "If you are **EXPLICITLY** instructed to **RELEASE** an object, set the gripper position to 0.0. Always set max_effort to 0.01.\n"
                "If you are instructed to move a certain direction (e.g., UP, DOWN, FORWARD, BACKWARD, LEFT, RIGHT), use the move_relative tool with small increments (e.g., 0.1m).\n"
                "If you are instructed to move to position you don't know, make reasonable assumptions, DO NOT ask for clarification.\n"
                "When you choose to use a tool, call it with appropriate arguments (if any). "
                "Return a final, concise, actionable response after using tools.\n"
                "Here is environment guidance:\n"
                "- Your gripper is currently at fully close position (0.8).\n"
                "- You can get your current gripper cartesian position by using the get_current_pose tool.\n"
                "- The direction up is along positive Z axis, down is along negative Z axis.\n"
                "- The direction forward is along positive X axis, backward is along negative X axis.\n"
                "- The direction left is along positive Y axis, right is along negative Y axis.\n"
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, max_iterations=10)

    # -----------------------
    # Action server callbacks
    # -----------------------
    def goal_callback(self, goal_request) -> GoalResponse:
        # Accept all goals (customize if needed)
        self.get_logger().info(f"[action] Received goal: {getattr(goal_request, 'prompt', '')}")
        # Reset tools_called for this execution
        with self._tools_called_lock:
            self._tools_called = []
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle) -> CancelResponse:
        self.get_logger().info("[action] Cancel request received.")
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """
        Called when a Prompt action goal is accepted.
        We run the agent and periodically publish feedback (tools_called).
        """

        prompt_text = goal_handle.request.prompt
        self.get_logger().info(f"[action] Executing prompt: {prompt_text}")

        feedback_msg = Prompt.Feedback()

        # Run agent in a different thread to avoid blocking ROS spin
        result_container: Dict[str, Any] = {"success": False, "final_response": "Internal error"}

        def run_agent():
            try:
                # invoke the agent (LangChain AgentExecutor). This may call our @tool fns.
                agent_resp = self.agent_executor.invoke({"input": prompt_text})
                # LangChain usually returns a dict with "output" key for final text
                final_text = agent_resp.get("output") if isinstance(agent_resp, dict) else str(agent_resp)
                result_container["success"] = True
                result_container["final_response"] = final_text
            except Exception as e:
                result_container["success"] = False
                result_container["final_response"] = f"Agent error: {e}"

        agent_thread = threading.Thread(target=run_agent, daemon=True)
        agent_thread.start()

        # While the agent runs, publish feedback every 0.5s with the current tools_called
        while agent_thread.is_alive():
            with self._tools_called_lock:
                # copy to avoid race
                tools_snapshot = list(self._tools_called)
            feedback_msg.tools_called = tools_snapshot
            try:
                goal_handle.publish_feedback(feedback_msg)
            except Exception:
                # ignore if cannot publish
                pass
            time.sleep(0.5)  # cooperative yield for ROS2
        # final publish
        with self._tools_called_lock:
            tools_snapshot = list(self._tools_called)
        feedback_msg.tools_called = tools_snapshot
        try:
            goal_handle.publish_feedback(feedback_msg)
        except Exception:
            pass

        # Prepare and return result
        result_msg = Prompt.Result()
        result_msg.success = bool(result_container.get("success", False))
        result_msg.final_response = str(result_container.get("final_response", ""))

        goal_handle.succeed()
        self.get_logger().info(f"[action] Goal finished. success={result_msg.success}")
        return result_msg


def main(args=None):
    rclpy.init(args=args)
    node = Ros2LLMAgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Ros2 LLM Agent Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
