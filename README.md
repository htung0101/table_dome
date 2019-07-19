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


3. Dump and Check your record

4. Pack tfrecords
  1. python packtf.py




Trouble shooting:
1. Unable to register with master node
  make sure you have another window running "roscore"

