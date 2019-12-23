# Bebop-Autonomy-Vision

An autonomous completely vision-based Bebop drone. From Intro to Robotics (CSCI-3302) project.

This project consists of ROS-based autonomous CNN-Based Navigation (via Dronet) of a Bebop Quadrotor with SSD300 Object Detection and Semi-Direct Visual Odometry. The whole vision suites only requires the Bebop's camera to run, no other sensors on the drone. Additionally, the object detection uses the python `torch2trt` plugin, which is a PyTorch to TensorRT converter that runs optimized models faster than ever.

## Contributors

- [Ayush Khanal](https://github.com/jptboy/)
- [Rami Mouro](https://github.com/ramalamadingdong/)
- [Brian Nguyen](https://github.com/BrianNguyen214)
- [Matthew Strong](https://github.com/peasant98)

## Relevant Papers

- [Dronet Paper](http://rpg.ifi.uzh.ch/docs/RAL18_Loquercio.pdf)

- [SVO Paper](https://www.ifi.uzh.ch/dam/jcr:e9b12a61-5dc8-48d2-a5f6-bd8ab49d1986/ICRA14_Forster.pdf)

- [SSD Object Detection Paper](https://arxiv.org/pdf/1512.02325.pdf)

## Package Description

- `bebop_autonomy` - The ROS driver for the Parrot Bebop drone. Is the base of this whole project. A link can be found [here](https://github.com/AutonomyLab/bebop_autonomy)

- `catkin_simple` - Catkin Simple ROS package that is used with Dronet. Additionally, a link to the Github repo can be found [here](https://github.com/catkin/catkin_simple).

- `dronet_control` - ETH Zurich's Dronet control package for sending commands to the `cmd_vel` for the Bebop drone. A link to all of Dronet can be found [here](https://github.com/uzh-rpg/rpg_public_dronet)

- `dronet_perception` - Runs the actual Dronet model (best if done on GPU), which outputs a steering angle and collision probability.

- `rpg_svo` - The semi-direct visual odometry package in ROS. Developed at ETH Zurich; the repository can be found [here](https://github.com/uzh-rpg/rpg_svo)

- `rpg_vikit` - Some vision tools for this project. The link to the repo is [here](https://github.com/uzh-rpg/rpg_vikit).

## Installation

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

## Misc Installation

We also have two Python files in this repo that are used for easier ROS control and for the object detection model, which includes `csci_dronet.py` and `robotics.py`.

- `csci_dronet.py` - This Python file serves as an easy way to send publish (in ROS) to one of three topics. Requires `argparse`, `std_msgs`, and `rospy`, if you haven't installed them already. Usage will be detailed in a later section.

- `robotics.py` - This Python file runs the SSD-300 object detection model in real time. Its usage will also be detailed in a later section. This requires:
  - `torch` - link [here](https://pytorch.org/get-started/locally/)
  - `torchvision`
  - `ros_numpy`- link [here](https://github.com/eric-wieser/ros_numpy)
  - `torch2trt` - link, as well as good installation steps, [here](https://github.com/NVIDIA-AI-IOT/torch2trt)

## Building the Code

- `cd bebop_ws`
- `catkin_make`
- If you have the above packages/dependencies installed, then `catkin_make` should work fine, however, it is possible that there is some missing ROS package (if there's an error). In that case, a common way to fix this issue is to run `sudo apt-get install ros-<your-distro>-<package-name>`. Then, retry the previous command.

Also do:

- In `bebop_ws/devel/include`: `sudo ln -s /opt/ros/<ros-distro>/include/parrot_arsdk parrot_arsdk`

- In `bebop_ws/devel/library`: `sudo ln -s /opt/ros/<ros-distro>/lib/parrot_arsdk parrot_arsdk`

- In your `~/.bashrc` file, add the following to the end:
`export LD_LIBRARY_PATH=<path-to-bebop-ws>/devel/lib/parrot_arsdk:$LD_LIBRARY_PATH`

## Steps to Running the Code

We explain how to use this open sourced work and how to integrate it with our work together.

First, make sure that you have a working Bebop2, and connect to its Wi-Fi network. Also, make sure that you successfully completed the build steps listed above.

- `cd bebop_ws`

- `source devel/setup.bash`

- `cd src/dronet_perception/launch`

- `roslaunch full_perception_launch.launch`

In another terminal, or using `tmux` (recommended):

- `cd bebop_ws/`

- `source devel/setup.bash`

- `cd src/dronet_control/launch`

- `roslaunch deep_navigation.launch`

### Object Detection

- Using `tmux` or another terminal:

- `python robotics.py` `python3.5 robotics.py`

- This should open a window that shows the detections from the Bebop drone in real time. Having a GPU helps here.

Here's an example:

![alt text](https://udana-documentation.s3-us-west-1.amazonaws.com/pictures/Screenshot+from+2019-12-18+13-49-48.png "SSD300")

### SVO

- Again, using `tmux` or another terminal:

`cd bebop_ws`
`source devel/setup.bash`
`roslaunch svo_ros live.launch`

For visualization:

`rosrun rviz rviz -d bebop_ws/src/rpg_svo/svo_ros/rviz_config.rviz`

![alt text](https://udana-documentation.s3-us-west-1.amazonaws.com/pictures/Screenshot+from+2019-12-18+12-20-06.png "SVO")

## Autonomous Navigation with Bebop

Now we can begin the real fun work. Below is a list of commands you can use once the above programs are running.

`rostopic pub --once /bebop/takeoff std_msgs/Empty` - takes off the drone

`rostopic pub --once /bebop/state_change std_msgs/Bool "data: true"` - enables dronet control. SVO and SSD will continue to run.

`rostopic pub --once /bebop/state_change std_msgs/Bool "data: false"` - stops dronet control, perception will still run.

`rostopic pub --once /bebop/land std_msgs/Empty` - lands the drone regardless of if dronet is enabled or not.

Additionally, we can run `python csci_dronet.py --option=takeoff`, `python csci_dronet.py --option=land`, `python csci_dronet.py --option=dronet_start`, or `python csci_dronet.py --option=dronet_end` to takeoff, land, start, and stop dronet, respectively.

## Real Life Vid/Pic

Dronet in action. It follows the road, and thankfully stops when one of us gets too close.  The Bebop doesn't listen to us, but Dronet's collision probability from its forward-facing camera was high enough so that the drone stopped, and disaster was averted.

<img src="dronet.gif?raw=true" width="1000px">

The convolutional neural network is influenced by edges (detailed more in the paper) and is clearly moving parallel to the edge of the road here.

![alt text](https://udana-documentation.s3-us-west-1.amazonaws.com/pictures/Screenshot+from+2019-12-22+23-04-46.png "Dronet")
