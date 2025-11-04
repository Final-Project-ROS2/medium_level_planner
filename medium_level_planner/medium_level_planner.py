import os
import threading
import time
from typing import List, Dict, Any, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, CancelResponse, GoalResponse
from rclpy.task import Future as RclpyFuture

# Services
from std_srvs.srv import SetBool

# Actions
from custom_interfaces.action import Prompt
from custom_interfaces.action import GetCurrentPose
from custom_interfaces.action import GetJointAngles
from custom_interfaces.action import MoveitRelative
from custom_interfaces.action import PlanComplexCartesianSteps
from control_msgs.action import GripperCommand
from geometry_msgs.msg import Pose

# LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, tool
from langchain.agents import AgentExecutor, create_tool_calling_agent

# dotenv
from dotenv import load_dotenv

ENV_PATH = '/home/group11/final_project_ws/src/medium_level_planner/.env'
load_dotenv(dotenv_path=ENV_PATH)

# ---------------------------
# Main Node
# ---------------------------
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

        # --- LangChain / LLM setup ---
        api_key = os.getenv("GEMINI_API_KEY") 
        if not api_key:
            self.get_logger().warn("No LLM API key found in environment variables GEMINI_API_KEY.")

        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.0)

        # Action clients
        self.move_action_client = ActionClient(self, PlanComplexCartesianSteps, "/plan_complex_cartesian_steps")
        self.pose_action_client = ActionClient(self, GetCurrentPose, "/get_current_pose")
        self.joint_action_client = ActionClient(self, GetJointAngles, "/get_joint_angles")
        self.relative_action_client = ActionClient(self, MoveitRelative, "/plan_cartesian_relative")

        if self.real_hardware:
            self.gripper_client = self.create_client(SetBool, "/control_gripper")
        else:
            self.gripper_client = ActionClient(self, GripperCommand, "/gripper_wrapper")

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

    # -----------------------
    # Tool wrappers (LangChain)
    # -----------------------
    def _initialize_tools(self) -> List[BaseTool]:
        """
        Define tools as @tool-decorated functions that internally call ROS service/action clients.
        Each tool appends its name to self._tools_called (thread-safe) so we can report feedback.
        """

        tools: List[BaseTool] = []

        # Action tools
        # /get_current_pose
        @tool
        def get_current_pose() -> str:
            """
            Returns the current robot's end-effector pose relative to base_link using an action call.
            """
            tool_name = "get_current_pose"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)

            try:
                if not self.pose_action_client.wait_for_server(timeout_sec=5.0):
                    return "Action server /get_current_pose unavailable"
                goal = GetCurrentPose.Goal()  # No goal fields required here
                send_future = self.pose_action_client.send_goal_async(goal)
                rclpy.spin_until_future_complete(self, send_future)
                goal_handle = send_future.result()
                if not goal_handle.accepted:
                    return "Action goal rejected for /get_current_pose"
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future)
                result = result_future.result().result
                if result.success:
                    return f"pose: {result.pose}"
                else:
                    return "Failed to get current pose"
            except Exception as e:
                return f"ERROR in {tool_name}: {e}"

        tools.append(get_current_pose)

        # /get_joint_angles
        @tool
        def get_joint_angles() -> str:
            """
            Returns a textual representation of all joint angles using an action call.
            """
            tool_name = "get_joint_angles"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)

            try:
                if not self.joints_action_client.wait_for_server(timeout_sec=5.0):
                    return "Action server /get_joint_angles unavailable"
                goal = GetJointAngles.Goal()
                send_future = self.joints_action_client.send_goal_async(goal)
                rclpy.spin_until_future_complete(self, send_future)
                goal_handle = send_future.result()
                if not goal_handle.accepted:
                    return "Action goal rejected for /get_joint_angles"
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future)
                result = result_future.result().result
                if result.success:
                    return f"joints: {result.joint_positions}"
                else:
                    return "Failed to get joint angles"
            except Exception as e:
                return f"ERROR in {tool_name}: {e}"


        # ---- Example Action tool: move_to_pose ----
        @tool
        def move_linear_to_pose(pos_x: float, pos_y: float, pos_z: float, rot_x: float, rot_y: float, rot_z: float, rot_w: float) -> str:
            """
            Move robot to a target pose using Roll Pitch Yaw orientation
            """
            tool_name = "move_linear_to_pose"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)

            try:
                # Example synchronous flow using ActionClient:
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
                return f"move_to_pose result: success={result.success}"
            except Exception as e:
                return f"ERROR in {tool_name}: {e}"

        tools.append(move_linear_to_pose)


        @tool
        def set_gripper_position(position: float, max_effort: float) -> str:
            """
            Set the gripper to a specific position with given max effort. Where a position of 0.0 is fully open and 0.8 is fully close.
            """
            tool_name = "set_gripper_position"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)

            try:
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
                return f"set_gripper_position result: success={result.reached_goal}"
            except Exception as e:
                return f"ERROR in {tool_name}: {e}"

        @tool
        def close_gripper(close: bool) -> str:
            """
            Close or open the gripper using a service call.
            Args:
                close (bool): True to close the gripper, False to open.
            """
            tool_name = "close_gripper"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)

            try:
                if not self.gripper_client.wait_for_service(timeout_sec=5.0):
                    return "Gripper service /control_gripper unavailable"
                request = SetBool.Request()
                request.data = close
                future = self.gripper_client.call_async(request)
                rclpy.spin_until_future_complete(self, future)
                response = future.result()
                if response.success:
                    action = "closed" if close else "opened"
                    return f"Gripper successfully {action}."
                else:
                    return f"Failed to {'close' if close else 'open'} gripper: {response.message}"
            except Exception as e:
                return f"ERROR in {tool_name}: {e}"
        
        if self.real_hardware:
            tools.append(close_gripper)
        else:
            tools.append(set_gripper_position)


        @tool
        def move_relative(dx: float, dy: float, dz: float,
                        roll: float, pitch: float, yaw: float) -> str:
            """
            Moves the robot end-effector relative to its current pose using the MoveItRelative action.

            Args:
                dx (float): Relative movement in x (meters)
                dy (float): Relative movement in y (meters)
                dz (float): Relative movement in z (meters)
                roll (float): Relative rotation in roll (radians)
                pitch (float): Relative rotation in pitch (radians)
                yaw (float): Relative rotation in yaw (radians)

            Returns:
                str: Result message indicating success or failure.
            """
            tool_name = "move_relative"
            with self._tools_called_lock:
                self._tools_called.append(tool_name)

            try:
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
                    return (f"✅ Relative motion executed successfully:\n"
                            f"Δx={dx:.3f}, Δy={dy:.3f}, Δz={dz:.3f}, "
                            f"roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")
                else:
                    return "❌ Cartesian path planning or execution failed."

            except Exception as e:
                return f"ERROR in {tool_name}: {e}"
        
        tools.append(move_relative)
            
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
                "query sensors or command the robot: get_current_pose, get_joint_angles, move_linear_to_pose, close_gripper, move_relative\n"
                "When you choose to use a tool, call it with appropriate arguments (if any). "
                "Return a final, concise, actionable response after using tools."
            )
        else:
            system_message = (
                "You are a ROS2-capable assistant. You can call the following tools (services/actions) to "
                "query sensors or command the robot: get_current_pose, get_joint_angles, move_linear_to_pose, set_gripper_position, move_relative\n"
                "If you are **EXPLICITLY** instructed to **GRAB** an object, simple set the gripper position to 0.2\n"
                "If you are **EXPLICITLY** instructed to **RELEASE** an object, simple set the gripper position to 0.0. Always set max_effort to 0.01\n"
                "If you are instructed to open or close the gripper without specific positions, assume fully open (0.0) or fully close (0.8) respectively.\n"
                "When you choose to use a tool, call it with appropriate arguments (if any). "
                "Return a final, concise, actionable response after using tools."
                "Here are some information about the environment:\n"
                "- Your gripper is currently at fully close position (0.8).\n"
                "- You can get your current gripper cartesian position by using the get_current_pose tool.\n"
                "- The direction up is along positive Z axis, down is along negative Z axis.\n"
                "- The direction forward is along positive X axis, backward is along negative X axis.\n"
                "- The direction left is along positive Y axis, right is along negative Y axis.\n"
                "- To orient the gripper to face downwards, set the orientation quaternion to (-0.69, 0.72, 0.00, 0.00).\n"
                "- The human operator you are assisting is to your right side (negative Y direction).\n"
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

# -----------------------
# Entrypoint
# -----------------------
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
