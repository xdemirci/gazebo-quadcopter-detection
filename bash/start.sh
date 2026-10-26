#/bin/bash
clear
x-terminal-emulator -e sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --console -I0 --sysid=101 --mcast --add-param-file=$HOME/gz_ws/src/ardupilot_gazebo/config/gazebo-iris-gimbal.parm -w
sleep 3
x-terminal-emulator -e sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --console -I1 --sysid=102 --mcast
sleep 3
gz sim -r iris_runway_two_quad.sdf
