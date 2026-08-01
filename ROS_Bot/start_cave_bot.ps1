# ─────────────────────────────────────────────────────────────
#  Cave Bot — Windows Launcher
#  Opens 4 SSH tabs in Windows Terminal, one per component.
#  Run: Right-click → "Run with PowerShell"
#       or:  powershell -ExecutionPolicy Bypass -File start_cave_bot.ps1
# ─────────────────────────────────────────────────────────────

$RPI   = "dharani@192.168.0.110"
$PASS  = "dharani!@#$"
$ROS   = "source /opt/ros/humble/setup.bash"
$SLAM_CFG = "`$HOME/cave_bot_slam.yaml"

# Commands to run on the RPi for each tab
$cmds = @{
    "LiDAR"      = "$ROS && ros2 launch ydlidar_ros2_driver ydlidar_launch.py"
    "Bridge"     = "sleep 3 && $ROS && python3 `$HOME/bot_script/real_bot.py"
    "SLAM"       = "sleep 5 && $ROS && ros2 launch slam_toolbox online_async_launch.py slam_params_file:=$SLAM_CFG"
    "Foxglove"   = "sleep 6 && $ROS && ros2 launch rosbridge_server rosbridge_websocket_launch.xml && $ROS && ros2 run tf2_ros static_transform_publisher 0 0 0.1 0 0 0 base_link laser_frame"
}

# Check if Windows Terminal (wt) is available
$wtAvailable = Get-Command wt -ErrorAction SilentlyContinue

if ($wtAvailable) {
    # Build a single `wt` command with multiple tabs
    $wtArgs = @()
    $first  = $true
    foreach ($tab in $cmds.GetEnumerator()) {
        $sshCmd = "plink -ssh -pw `"$PASS`" $RPI `"$($tab.Value); exec bash`""
        if ($first) {
            $wtArgs += "new-tab --title `"$($tab.Key)`" -- powershell -NoExit -Command `"$sshCmd`""
            $first = $false
        } else {
            $wtArgs += "; new-tab --title `"$($tab.Key)`" -- powershell -NoExit -Command `"$sshCmd`""
        }
    }
    Start-Process wt -ArgumentList ($wtArgs -join " ")
    Write-Host "✅ Opened Windows Terminal with 4 tabs." -ForegroundColor Green
} else {
    # Fallback: open 4 separate PowerShell windows
    Write-Host "Windows Terminal not found — opening separate windows..." -ForegroundColor Yellow
    foreach ($tab in $cmds.GetEnumerator()) {
        $sshCmd = "plink -ssh -pw '$PASS' $RPI '$($tab.Value); exec bash'"
        Start-Process powershell -ArgumentList "-NoExit", "-Command", $sshCmd `
            -WindowStyle Normal
        Start-Sleep -Milliseconds 500
    }
    Write-Host "✅ Opened 4 PowerShell windows." -ForegroundColor Green
}

Write-Host ""
Write-Host "Tabs / Windows:" -ForegroundColor Cyan
Write-Host "  [LiDAR]    - YDLidar driver"
Write-Host "  [Bridge]   - real_bot.py (Arduino bridge + IMU)"
Write-Host "  [SLAM]     - slam_toolbox"
Write-Host "  [Foxglove] - rosbridge + static TF"
