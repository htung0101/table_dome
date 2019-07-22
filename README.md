Users:

by "config file" I mean config.py

0. set up hardware and go to the exp folder

  1.in the terminal, call

   source goto_table_dome.sh
   table_dome
   source source_this.py

   to come to this folder
  2. Open another window running "roscore"
     * if it says there is already some running roscore, then don't run it

  3. Open another terminal, do:

  source /home/zhouxian/catkin_ws/devel/setup.bash
  roslaunch realsense2_camera rs_aligned_depth_multiple_cameras.launch camera1:=camera1 serial_nocamera1:=836612072253 camera2:=camera2 serial_nocamera2:=838212071161 camera3:=camera3 serial_nocamera3:=838212071165 camera4:=camera4 serial_nocamera4:=831612072676 camera5:=camera5 serial_nocamera5:=826212070528 camera6:=camera6 serial_nocamera6:=838212071158

  make sure you get some yellow messages saying [WARN][...]Hardware x, Error
  if you can something white, please unplug the camera usb and insert again



1. Calibrate
   checkerboard calibrate
   0. source /home/zhouxian/kalibr_workspace/devel/setup.bash
   1. python3 user_scripts/checkerboard_calibrate.py
      a. There will be a countdown.
      b. Start recording once the countdown is finished 5,4,3,2,1

      c. End recording until the terminal prints "finish anytime you want"
      with ctrl-c
      d. follow the instruction in the end of the message, by
         * adding checkboard_record_name to config, and
         * running kalibr_calibrate_cameras command

   note: this one giveS me the CamX_T_CamY

   2. After finish it, place the yaml to your target folder: checkerboard_data_root/checkerboard_record_name in config.py
      cd /home/zhouxian/data/TableDome/cam_calibrate/TableDome_you_exp_name
      and add your yml file to config: checkboard_yml=


*****
   Artag_calibration
   0. put the ar tag on the table (with the correct orientation)
   1. python3 user_scripts/calibrate.py

   note: this will create a folder ~/data/TableDome/record_name
   and store artag_T_cam and camera intrinsics under ar_tag folder


2. Record
   0. put away the ar tag
   1. remember to put your record name and artag_folder in the config.py file
   2. python3 user_scripts/record.py
   3. kill the process with ctrl-c once you want to finish recording


   4. If you want to record without recalibrate, keep the same ar_tag_folder in config.py,
      and change your record name to record_name=""

3. Check your record
  1. rqt_bag /home/zhouxian/data/TableDome/YOUR_RECORD_NAME/CalibData.bag

4. Pack tfrecords:
  1. python3 user_scripts/dump.py
  1. python packtf.py

  Note: save bag into image/depth, then pack everything with packtf.py
****

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