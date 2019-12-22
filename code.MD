# CSCI 3002: Intro to Robotics Final Project

Autonomous CNN-Based Navigation of a Quadrotor with SSD300 Object Detection and Semi-Direct Visual Odometry

We have attached the two separate ROS workspaces - `svo_ws`, `bebop_ws`, and the supplementary python, and this file.

Within `svo_ws`, we used the two open sourced packages `rpg_svo` and `rpg_vikit`.
Within `bebop_ws`, we used `bebop_autonomy`, `catkin_simple ` (the old Dronet worked on old versions of ROS), `dronet_control`, and `dronet_perception`
We explain how to used this open sourced work and how to integrate it with our work together.

Requires:

`bebop_autonomy`
ETH Zurich's `dronet` ros packages from [here](https://github.com/uzh-rpg/rpg_public_dronet/)
ETH Zurich's `svo` ros packages from [here](https://github.com/uzh-rpg/rpg_svo)
`torch2trt`
`tensorflow`
`keras`
`pytorch`

## Steps

We used two ROS workspaces for getting everything up and running - `svo_ws/` and `bebop_ws`.
In `svo_ws/`, we have the ROS SVO packages. In the `bebop_ws` package, we have `bebop_autonomy`, and the
`dronet` packages there. This included `dronet_perception` and `dronet_control` for `cpp` control.


## Steps to Running the Code
`cd bebop_ws/`
`source devel/setup.bash`
`cd src/dronet_perception/launch`
`roslaunch full_perception_launch.launch`

In another terminal:

`cd bebop_ws/`
`source devel/setup.bash`
`cd src/dronet_control/launch`
`roslaunch deep_navigation.launch`

### Object Detection:
run `python3.5 robotics.py` after.

Requires Tensorflow, Keras, Torch2TRT (torch to TensorRT), rospy

should display something like one of the images.

### SVO
`cd svo_ws`
`source devel/setup.bash`
`roslaunch svo_ros live.launch`

visualization:

`rosrun rviz rviz -d <PATH-TO-SVO-WS>/src/g_svo/svo_ros/rviz_config.rviz`

## Real Time Work

Now we can begin the real fun work.

`rostopic pub --once /bebop/takeoff std_msgs/Empty` - takes off the drone

`rostopic pub --once /bebop/state_change std_msgs/Bool "data: true"` - enables dronet control. SVO and SSD will continue to run.

`rostopic pub --once /bebop/state_change std_msgs/Bool "data: false"` - stops dronet control, perception will still run.

`rostopic pub --once /bebop/land std_msgs/Empty` - lands the drone regardless of if dronet is enabled or not.

Additionally, we can run `python csci_dronet.py --option=takeoff`, `python csci_dronet.py --option=land`, `python csci_dronet.py --option=dronet_start`, or `python csci_dronet.py --option=dronet_end` to takeoff, land, start, and stop dronet, respectively.

We also have attached some of the dronet code from ETH Zurich RPG code that we were able to get integrated with everything.
