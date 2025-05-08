"""Functions to convert between ROS messages to numpy arrays."""

import numpy as np
from rosbags.image import message_to_cvimage
from scipy.spatial.transform import Rotation as R


def ros_img_to_np(ros_img, encoding: str = "bgr8") -> np.ndarray:
    return np.asarray(message_to_cvimage(ros_img, encoding))


def ros_array_to_np(ros_array) -> np.ndarray:
    return np.array(ros_array.data, dtype=np.float32)


def ros_pose_to_np(ros_pose) -> np.ndarray:
    translation = np.array(
        [ros_pose.position.x, ros_pose.position.y, ros_pose.position.z],
    )
    rotation = np.array(
        [
            ros_pose.orientation.x,
            ros_pose.orientation.y,
            ros_pose.orientation.z,
            ros_pose.orientation.w,
        ],
    )
    rotation_matrix = R.from_quat(rotation).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    return transform


def ros_pose_stamped_to_np(ros_pose_stamped) -> np.ndarray:
    return ros_pose_to_np(ros_pose_stamped.pose)


def convert_ros_msg_to_np(ros_msg, msg_type) -> np.ndarray:
    if msg_type == "std_msgs/msg/Float32MultiArray":
        return ros_array_to_np(ros_msg)
    elif msg_type == "sensor_msgs/msg/Image":
        return ros_img_to_np(ros_msg)
    elif msg_type == "geometry_msgs/msg/PoseStamped":
        return ros_pose_stamped_to_np(ros_msg)
    else:
        raise ValueError(f"Unknown message type: {msg_type}")
