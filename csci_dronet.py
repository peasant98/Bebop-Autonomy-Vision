import rospy
from std_msgs.msg import Empty, Bool

import argparse

# quick python script to send the only controls you need for the bebop drone to fly.
# requires bebop autonomy ros to already be running

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str, default='land', help='season year')
    rospy.init_node('bebop_dronet_control')
    # takeoff, dronet-switch, landing controller
    takeoff_pub = rospy.Publisher('bebop/takeoff', Empty)
    control_pub = rospy.Publisher('bebop/state_change', Bool)
    land_pub = rospy.Publisher('bebop/land', Empty)
    opt = parser.parse_args()
    selection = opt.option
    if selection == 'takeoff':
        # takeoff drone
        e = Empty()
        takeoff_pub.publish(e)
    elif selection == 'dronet_start':
        # turn on dronet control
        b = Bool()
        b.data = True
        control_pub.publish(b)
        pass
    elif selection == 'dronet_end':
        # turn off dronet control
        b = Bool()
        b.data = False
        control_pub.publish(b)
    # if any other command is pressed, simply land
    else:
        e = Empty()
        land_pub.publish(e)