from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sim_arg = DeclareLaunchArgument(
        'sim',
        default_value='false',
        description='Use the simulator IP address when true',
    )

    ip_arg = DeclareLaunchArgument(
        'ip',
        default_value=PythonExpression([
            "'10.202.0.1' if '",
            LaunchConfiguration('sim'),
            "'.lower() in ('true', '1', 'yes') else '192.168.53.1'",
        ]),
        description='Anafi drone IP address',
    )

    model_arg = DeclareLaunchArgument(
        'model',
        default_value='4k',
        description='Anafi drone model',
    )

    anafi_autonomy = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('anafi_autonomy'),
                'launch',
                'anafi_autonomy_launch.py',
            ])
        ),
        launch_arguments={
            'ip': LaunchConfiguration('ip'),
            'model': LaunchConfiguration('model'),
        }.items(),
    )

    local_odom = Node(
        package='control',
        executable='anafi_local_odom_node',
        name='anafi_local_odom_node',
        output='screen',
    )

    return LaunchDescription([
        sim_arg,
        ip_arg,
        model_arg,
        anafi_autonomy,
        local_odom,
    ])
