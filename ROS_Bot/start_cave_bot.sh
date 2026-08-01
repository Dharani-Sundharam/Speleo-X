#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  Cave Bot — Full System Startup
#  Usage: ./start_cave_bot.sh
#  Requires: tmux  (sudo apt install tmux)
# ─────────────────────────────────────────────────────────────

SESSION="cave_bot"

# ── Update this when the laptop's IP changes ──────────────────
LAPTOP_IP="10.17.144.64"      # laptop IP for camera stream (UDP port 5000)

# Enable mouse mode so you can click between tmux windows/panes
TMUX_MOUSE="set -g mouse on"
SLAM_CONFIG="$HOME/cave_bot_slam.yaml"
BRIDGE_SCRIPT="$HOME/bot_script/real_bot.py"

# Source ROS 2 in every new window
ROS_SOURCE="source /opt/ros/humble/setup.bash"

# Kill any previous session cleanly
tmux kill-session -t $SESSION 2>/dev/null

echo "Starting Cave Bot..."

# ── Window 0: Flat 2D LiDAR ────────────────────────────────
tmux new-session -d -s $SESSION -n "lidar_base"
tmux set-option -t $SESSION mouse on   # click between windows with the mouse
tmux send-keys -t $SESSION:lidar_base \
  "$ROS_SOURCE && source ~/ydlidar_ws/install/setup.bash && ros2 launch ydlidar_ros2_driver ydlidar_launch.py" Enter

sleep 8  # Give the first LiDAR motor time to spin up and stabilize power draw

# ── Window 1: Tilted 3D LiDAR ──────────────────────────────
tmux new-window -t $SESSION -n "lidar_tilted"
tmux send-keys -t $SESSION:lidar_tilted \
  "$ROS_SOURCE && source ~/ydlidar_ws/install/setup.bash && ros2 launch ydlidar_ros2_driver ydlidar_tilted_launch.py" Enter

# ── Window 2: Arduino Bridge (real_bot.py) ───────────────────
tmux new-window -t $SESSION -n "bridge"
tmux send-keys -t $SESSION:bridge \
  "$ROS_SOURCE && python3 $BRIDGE_SCRIPT" Enter

sleep 1

# ── Window 2: SLAM Toolbox (2D SLAM) ──────────────────────────
tmux new-window -t $SESSION -n "slam"
tmux send-keys -t $SESSION:slam \
  "$ROS_SOURCE && ros2 launch slam_toolbox online_async_launch.py \
   slam_params_file:=$SLAM_CONFIG" Enter

# ── Window 3: Robot State Publisher (URDF) ───────────────────
tmux new-window -t $SESSION -n "urdf"
tmux send-keys -t $SESSION:urdf \
  "$ROS_SOURCE && ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:=\"\$(cat $HOME/bot_script/cave_bot.urdf)\"" Enter

# ── Window 4: Nav2 (disabled — mapping-only mode) ────────────
# Uncomment to re-enable autonomous navigation:
# tmux new-window -t $SESSION -n "nav2"
# tmux send-keys -t $SESSION:nav2 \
#   "$ROS_SOURCE && ros2 launch nav2_bringup navigation_launch.py \
#    params_file:=$HOME/cave_bot_nav2.yaml use_sim_time:=false" Enter

# ── Window 5: 3D Push-Broom Point Cloud (tilted LiDAR) ──────
tmux new-window -t $SESSION -n "3d_map"
tmux send-keys -t $SESSION:3d_map \
  "$ROS_SOURCE && python3 $HOME/bot_script/laser_stitcher.py" Enter

# ── Window 5: Map Save (run commands here when ready) ───────
tmux new-window -t $SESSION -n "map_save"
tmux send-keys -t $SESSION:map_save \
  "$ROS_SOURCE && echo 'RTAB-Map saves automatically to ~/cave_map.db on shutdown.' && echo 'To save NOW: ros2 service call /rtabmap/backup_database std_srvs/srv/Empty' && echo 'To export PCD: rtabmap-databaseViewer ~/cave_map.db'" Enter

# ── Window: Camera Stream (CSI → Laptop via UDP) ─────────────
tmux new-window -t $SESSION -n "camera"
tmux send-keys -t $SESSION:camera \
  "gst-launch-1.0 -e \
   v4l2src device=/dev/video0 ! \
   video/x-raw,width=640,height=480,framerate=30/1 ! \
   videoconvert ! video/x-raw,format=I420 ! \
   x264enc bitrate=4000 tune=zerolatency speed-preset=ultrafast key-int-max=30 ! \
   mpegtsmux ! \
   udpsink host=${LAPTOP_IP} port=5000 sync=false async=false" Enter

# ── Window 7: Foxglove Bridge (Port 8765) ────────────────────
tmux new-window -t $SESSION -n "foxglove"
tmux send-keys -t $SESSION:foxglove \
  "$ROS_SOURCE && ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765" Enter

# ── Done ─────────────────────────────────────────────────────
echo ""
echo "✅ All systems launched in tmux session: '$SESSION'"
echo ""
echo "   Attach to see all windows:  tmux attach -t $SESSION"
echo "   Switch windows:             Ctrl+B then 0-4"
echo "   Detach (leave running):     Ctrl+B then D"
echo "   Kill everything:            tmux kill-session -t $SESSION"
echo ""
