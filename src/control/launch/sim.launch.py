"""
simulation.launch.py
════════════════════
Launches the Gazebo Sim stack in surveillance_building.world and spawns
the drone model from this package:
    1. Gazebo Sim       +  surveillance_building.world
    2. model.sdf        (spawned as my_vehicle)
    3. Camera bridge    Gazebo Sim → /anafi/image topics
    4. Static TF        world → odom

Usage
─────
  ros2 launch control simulation.launch.py

Optional arguments
──────────────────
        world:=<path>      Override world file path
        world_name:=...    Gazebo world name for spawning
        file:=<path>       Override model.sdf path
        entity_name:=...   Spawned entity name
        x:=5.0             Model spawn X
        y:=5.0             Model spawn Y
        z:=0.5             Model spawn Z
    yaw:=-1.5708       Model spawn yaw in radians

Spawn the drone manually with the same defaults
───────────────────────────────────────
    ros2 run ros_gz_sim create -world surveillance_building -file <path>/model.sdf \
        -name {vehicle_name} -x 5.0 -y 5.0 -z 0.5 -Y -1.5708

View the bridged camera in rqt
──────────────────────────
    ros2 run rqt_image_view rqt_image_view /{vehicle_name}/image

"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
    LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # ── Package paths ─────────────────────────────────────────────────
    pkg_this = get_package_share_directory('control')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world_file_default = os.path.join(pkg_this, 'simulator', 'worlds', 'surveillance_building.world')
    model_file_default = os.path.join(pkg_this, 'simulator', 'models', 'model.sdf')
    vehicle_name = 'anafi4k'

    bridge_config = os.path.join(
        pkg_this,
        'simulator',
        'config',
        'bridges.yaml'
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[
            {'config_file': bridge_config}
        ],
        output='screen'
    )

    # ── Declare launch arguments ──────────────────────────────────────
    args = [
        DeclareLaunchArgument('world', default_value=world_file_default,
                              description='Path to world file'),
        DeclareLaunchArgument('world_name', default_value='surveillance_building',
                              description='Gazebo world name'),
        DeclareLaunchArgument('file', default_value=model_file_default,
                              description='Path to model.sdf file'),
        DeclareLaunchArgument('entity_name', default_value=vehicle_name,
                              description='Spawned entity name'),
        DeclareLaunchArgument('x', default_value='0.0',
                              description='Model spawn X position'),
        DeclareLaunchArgument('y', default_value='5.0',
                              description='Model spawn Y position'),
        DeclareLaunchArgument('z', default_value='0.5',
                              description='Model spawn Z position'),
        DeclareLaunchArgument('yaw', default_value='-1.5708',
                      description='Model spawn yaw in radians'),
    ]

    # ── 1. Launch Gazebo Sim with the empty world ────────────────────
    use_software_rendering = [
        SetEnvironmentVariable(name='LIBGL_ALWAYS_SOFTWARE', value='0'),
        SetEnvironmentVariable(name='MESA_LOADER_DRIVER_OVERRIDE', value='llvmpipe'),
    ]

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': ['-r ', LaunchConfiguration('world')],
        }.items(),
    )

    # ── 2. Spawn the local model.sdf into the running world ──────────
    spawn_model = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-world', LaunchConfiguration('world_name'),
            '-file', LaunchConfiguration('file'),
            '-name', LaunchConfiguration('entity_name'),
            '-x', LaunchConfiguration('x'),
            '-y', LaunchConfiguration('y'),
            '-z', LaunchConfiguration('z'),
            '-Y', LaunchConfiguration('yaw'),
        ],
        output='screen',
    )

    delayed_spawn = TimerAction(period=3.0, actions=[spawn_model])

    # ── 3. Bridge topics from Gazebo Sim into ROS 2 ──────────
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[
            {'config_file': bridge_config}
        ],
        output='screen'
    )

    delayed_bridge = TimerAction(
        period=5.0,
        actions=[bridge]
    )

    # ── 4. Static TFs: ────────────────────────────────────
    world_to_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_odom_tf',
        arguments=['0', '0', '0',   # x y z
                   '0', '0', '0',   # roll pitch yaw
                   'world', 'odom'],
        output='screen',
    )

    camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_tf',
        arguments=[
            '0.05', '0', '0.02',   # x y z (adjust if needed)
            '0', '0', '0',         # roll pitch yaw
            'anafi4k/base_link',   # parent (matches your camera_info)
            'camera_link'
        ],
        output='screen',
    )

    camera_optical_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_to_optical_tf',
        arguments=[
            '0', '0', '0',
            '-1.5708', '0', '-1.5708',  # standard ROS camera optical rotation
            'camera_link',
            'camera_optical_frame'
        ],
        output='screen',
    )

    # ── 5. Gazebo to ROS odometry ────────────────────────────────────────────
    gazebo_pose_to_odom = Node(
        package='control',
        executable='gazebo_pose_to_odom',
        name='gazebo_pose_to_odom',
        output='screen',
    )

    # ── 6. Log useful info ────────────────────────────────────────────
    info = [
        LogInfo(msg='─────────────────────────────────────────────'),
        LogInfo(msg='  World  : ' + world_file_default),
        LogInfo(msg='  Model  : ' + model_file_default),
        LogInfo(msg='  Entity : ' + vehicle_name),
        LogInfo(msg='─────────────────────────────────────────────'),
    ]

    return LaunchDescription(
        args + info + [
            gazebo,
            delayed_spawn,
            delayed_bridge,
            gazebo_pose_to_odom,
            world_to_odom_tf,
            camera_tf,
            camera_optical_tf
        ]
    )
