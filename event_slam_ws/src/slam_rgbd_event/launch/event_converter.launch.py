from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Your voxel grid node
        Node(
            package='slam_rgbd_event',
            executable='visualization_node',
            output='screen',
            parameters=[{'use_sim_time': True}],
        ),
        # Event camera renderer (visualization)
        Node(
            package='event_camera_renderer',
            executable='renderer_node',
            # name='event_renderer',
            parameters=[
                {'use_sim_time': True},
            ],
            output='screen',
            remappings=[
                ('~/events', '/event_camera/events'),
            ],
        ),
    ])
