Users:


0.1 in the terminal, call

   source goto_table_dome.sh
   table_dome

   to come to this folder

1. Calibrate (Done)
   0. put the ar tag on the table (with the correct orientation)
   1. python3 user_scripts/calibrate.py

   note: this will create a folder ~/data/TableDome/record_name
   and store vrtag_T_cam under vr_tag folder


2. Record
   0. put away the vr tag
   1. remember to put your record name in the config.py file
   2. python3 user_scripts/record.py
   3. kill the process once you want to finish recording


3. Check your record
  1. rqt_bag /home/zhouxian/data/TableDome/YOUR_RECORD_NAME/CalibData.bag

4. Pack tfrecords:
  1. python3 user_scripts/record.py
  1. python packtf.py

  Note: save bag into image/depth, then pack everything with packtf.py



Trouble shooting:
1. Unable to register with master node
  make sure you have another window running "roscore"


Note:
todo:
1. if you use wait_for_message, you don't need the subscriber

2. color and depth path should be /colorData/Cam%d/cam_camera%d_%d_color.npy
                                  /depthData/Cam%d/cam_camera%d_%d_depth.npy