# Bebop-Autonomy-Vision

An autonomous completely vision-based Bebop drone. From Intro to Robotics project.

This project consists of autonomous CNN-Based Navigation of a Bebop Quadrotor with SSD300 Object Detection and Semi-Direct Visual Odometry. The whole vision suites only requires the Bebop's camera to run, no other sensors on the drone.

## Contributors

- [Ayush Khanal](https://github.com/jptboy/)
- [Rami Mouro](https://github.com/ramalamadingdong/)
- [Brian Nguyen](https://github.com/BrianNguyen214)
- [Matthew Strong](https://github.com/peasant98)

### Package Description

- `bebop_autonomy` - The ROS driver for the Parrot Bebop drone. Is the base of this whole project.

- `catkin_simple` - Catkin Simple ROS package that is used with Dronet.

- `dronet_control` - Dronet control package for sending commands to the `cmd_vel` for the Bebop drone.

- `dronet_perception` - Runs the actual Dronet model (best if done on GPU), which outputs a steering angle and collision probability.

- `rpg_svo` - The semi-direct visual odometry package in ROS.

- `rpg_vikit` - Some vision tools for this project.

### Installation

Each package with the `bebop_ws` ROS workspace requires some different work to be done to get it fully working with the whole suite. They are listed below (firstly, make sure that you have ROS installed).

- `bebop_autonomy` - in-depth docs [here](https://bebop-autonomy.readthedocs.io/en/latest/)
  - Run `sudo apt-get install ros-<ros-distro>-parrot-arsdk`

- `catkin_simple` - simply required for being able to build Dronet, which uses `catkin_simple`

- `dronet_control` - ROS package that takes the CNN predictions from Dronet to send the correct commands to the Bebop drone.

- `dronet_perception` - ROS package that runs the actual Dronet model.
  - Requires `tensorflow`, `keras`, `rospy`, opencv, Python's gflags, and numpy/sklearn.
  - More information about the Dronet code and setup can be found [here](https://github.com/uzh-rpg/rpg_public_dronet).

- `rpg_svo` - There are some extra steps that you will need to follow; these are detailed well at `rpg_svo`'s [wiki](https://github.com/uzh-rpg/rpg_svo/wiki/Installation:-ROS). The `g2o` package is optional. Additionally, the step to clone `rpg_svo` is not needed as it already exists in this repo.

- `rgp_vikit` - Nothing here.

### Misc Installation

We also have two Python files in this repo that are used for easier ROS control and for the object detection model, which includes `csci_dronet.py` and `robotics.py`.

- `csci_dronet.py` - This Python file 

- `robotics.py` - This Python file

We have attached the two separate ROS workspaces - `svo_ws`, `bebop_ws`, and the supplementary python, and this file.

Within `svo_ws`, we used the two open sourced packages `rpg_svo` and `rpg_vikit`.
Within `bebop_ws`, we used `bebop_autonomy`, `catkin_simple ` (the old Dronet worked on old versions of ROS), `dronet_control`, and `dronet_perception`
We explain how to used this open sourced work and how to integrate it with our work together.



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
