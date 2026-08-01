#!/usr/bin/env python3
urdf = """<?xml version="1.0"?>
<robot name="cave_bot">
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.27 0.25 0.08"/></geometry>
      <material name="blue"><color rgba="0.2 0.4 0.8 0.8"/></material>
    </visual>
    <visual>
      <origin xyz="0.11 0 0.045" rpy="0 0 0"/>
      <geometry><box size="0.05 0.23 0.02"/></geometry>
      <material name="red"><color rgba="0.9 0.1 0.1 1.0"/></material>
    </visual>
    <visual>
      <origin xyz="0.08 0 0.065" rpy="0 0 0"/>
      <geometry><cylinder radius="0.035" length="0.05"/></geometry>
      <material name="black"><color rgba="0.1 0.1 0.1 1.0"/></material>
    </visual>
  </link>
  <link name="laser_frame"/>
  <link name="laser_tilted_frame"/>
  <joint name="laser_joint" type="fixed">
    <parent link="base_link"/>
    <child link="laser_frame"/>
    <origin xyz="0.10 0 0.15" rpy="0 0 0"/>
  </joint>
  <joint name="laser_tilted_joint" type="fixed">
    <parent link="base_link"/>
    <child link="laser_tilted_frame"/>
    <origin xyz="0.08 0 0.22" rpy="0 -1.047 0"/>
  </joint>
</robot>
"""
with open('/home/dharani/bot_script/cave_bot.urdf', 'w', newline='\n') as f:
    f.write(urdf)
print('URDF written with Unix line endings!')
