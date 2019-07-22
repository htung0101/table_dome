Users:


0. set up hardware and go to the exp folder

  1.in the terminal, call

   source goto_table_dome.sh
   table_dome

   to come to this folder
  2. make sure you have another window running "roscore"

  3. in a terminal, do:

  source /home/zhouxian/catkin_ws/devel/setup.bash
  roslaunch realsense2_camera rs_aligned_depth_multiple_cameras.launch camera1:=camera1 serial_nocamera1:=836612072253 camera2:=camera2 serial_nocamera2:=838212071161 camera3:=camera3 serial_nocamera3:=838212071165 camera4:=camera4 serial_nocamera4:=831612072676 camera5:=camera5 serial_nocamera5:=826212070528 camera6:=camera6 serial_nocamera6:=838212071158

  make sure you get some yellow messages saying [WARN][...]Hardware x, Error
  if you can something white, please unplug the camera usb and insert again



1. Calibrate
   checkerboard calibrate
   0. source /home/zhouxian/kalibr_workspace/devel/setup.bash
   1. python3 user_scrtips/checkerboard_calibrate.py
   note: this one giveS me the CamX_T_CamY

   artag_calibration
   0. put the ar tag on the table (with the correct orientation)
   1. python3 user_scripts/calibrate.py

   note: this will create a folder ~/data/TableDome/record_name
   and store vrtag_T_cam and camera intrinsics under vr_tag folder


2. Record
   0. put away the vr tag
   1. remember to put your record name in the config.py file
   2. python3 user_scripts/record.py
   3. kill the process once you want to finish recording


3. Check your record
  1. rqt_bag /home/zhouxian/data/TableDome/YOUR_RECORD_NAME/CalibData.bag

4. Pack tfrecords:
  1. python3 user_scripts/dump.py
  1. python packtf.py

  Note: save bag into image/depth, then pack everything with packtf.py



==========extra===============
1. visualize your point cloud with
    python user_scripts/visual.py





Trouble shooting:
1. Unable to register with master node
  make sure you have another window running "roscore"


Note:
todo:
1. if you use wait_for_message, you don't need the subscriber

2. color and depth path should be /colorData/Cam%d/cam_camera%d_%d_color.npy
                                  /depthData/Cam%d/cam_camera%d_%d_depth.npy