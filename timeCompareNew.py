import numpy as np
import yaml
import argparse
import os
import pickle
import ipdb
st=ipdb.set_trace

'''
    COMMAND FOR RUNNING THE CODE:
    
    python timeCompare.py --yaml_color1Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/colorData/Cam1/data.yaml --yaml_color2Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/colorData/Cam2/data.yaml --yaml_color3Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/colorData/Cam3/data.yaml --yaml_color4Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/colorData/Cam4/data.yaml --yaml_color5Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/colorData/Cam5/data.yaml --yaml_color6Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/colorData/Cam6/data.yaml --yaml_depth1Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/depthData/Cam1/data.yaml --yaml_depth2Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/depthData/Cam2/data.yaml --yaml_depth3Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/depthData/Cam3/data.yaml --yaml_depth4Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/depthData/Cam4/data.yaml --yaml_depth5Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/depthData/Cam5/data.yaml --yaml_depth6Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/depthData/Cam6/data.yaml

    python timeCompareNew.py --yaml_color1Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData4/colorData/Cam1/data.yaml --yaml_color2Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData4/colorData/Cam2/data.yaml --yaml_color3Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData4/colorData/Cam3/data.yaml --yaml_color4Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData4/colorData/Cam4/data.yaml --yaml_color5Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData4/colorData/Cam5/data.yaml --yaml_color6Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData4/colorData/Cam6/data.yaml --yaml_depth1Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData4/depthData/Cam1/data.yaml --yaml_depth2Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData4/depthData/Cam2/data.yaml --yaml_depth3Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData4/depthData/Cam3/data.yaml --yaml_depth4Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData4/depthData/Cam4/data.yaml --yaml_depth5Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData4/depthData/Cam5/data.yaml --yaml_depth6Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData4/depthData/Cam6/data.yaml --save_dir /media/zhouxian/Elements/Kalibrbackup/TrainData/newData4

    python timeCompareNew.py --yaml_color1Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h13_m52_s4/rgb_depth_npy/colorData/Cam1/data.yaml --yaml_color2Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h13_m52_s4/rgb_depth_npy/colorData/Cam2/data.yaml --yaml_color3Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h13_m52_s4/rgb_depth_npy/colorData/Cam3/data.yaml --yaml_color4Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h13_m52_s4/rgb_depth_npy/colorData/Cam4/data.yaml --yaml_color5Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h13_m52_s4/rgb_depth_npy/colorData/Cam5/data.yaml --yaml_color6Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h13_m52_s4/rgb_depth_npy/colorData/Cam6/data.yaml --yaml_depth1Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h13_m52_s4/rgb_depth_npy/depthData/Cam1/data.yaml --yaml_depth2Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h13_m52_s4/rgb_depth_npy/depthData/Cam2/data.yaml --yaml_depth3Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h13_m52_s4/rgb_depth_npy/depthData/Cam3/data.yaml --yaml_depth4Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h13_m52_s4/rgb_depth_npy/depthData/Cam4/data.yaml --yaml_depth5Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h13_m52_s4/rgb_depth_npy/depthData/Cam5/data.yaml --yaml_depth6Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h13_m52_s4/rgb_depth_npy/depthData/Cam6/data.yaml --save_dir /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h13_m52_s4/rgb_depth_npy

    python timeCompareNew.py --yaml_color1Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h18_m19_s19/rgb_depth_npy/colorData/Cam1/data.yaml --yaml_color2Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h18_m19_s19/rgb_depth_npy/colorData/Cam2/data.yaml --yaml_color3Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h18_m19_s19/rgb_depth_npy/colorData/Cam6/data.yaml --yaml_color4Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h18_m19_s19/rgb_depth_npy/colorData/Cam4/data.yaml --yaml_color5Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h18_m19_s19/rgb_depth_npy/colorData/Cam5/data.yaml --yaml_color6Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h18_m19_s19/rgb_depth_npy/colorData/Cam6/data.yaml --yaml_depth1Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h18_m19_s19/rgb_depth_npy/depthData/Cam1/data.yaml --yaml_depth2Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h18_m19_s19/rgb_depth_npy/depthData/Cam2/data.yaml --yaml_depth3Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h18_m19_s19/rgb_depth_npy/depthData/Cam6/data.yaml --yaml_depth4Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h18_m19_s19/rgb_depth_npy/depthData/Cam4/data.yaml --yaml_depth5Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h18_m19_s19/rgb_depth_npy/depthData/Cam5/data.yaml --yaml_depth6Path /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h18_m19_s19/rgb_depth_npy/depthData/Cam6/data.yaml --save_dir /home/zhouxian/data/TableDome/artag_only_TableDome_y2019_m7_h18_m19_s19/rgb_depth_npy

    python timeCompare.py --yaml_color1Path /home/zhouxian/catkin_ws/newData2/colorData/Cam1/data.yaml --yaml_color2Path /home/zhouxian/catkin_ws/newData2/colorData/Cam2/data.yaml --yaml_color3Path /home/zhouxian/catkin_ws/newData2/colorData/Cam3/data.yaml --yaml_color4Path /home/zhouxian/catkin_ws/newData2/colorData/Cam4/data.yaml --yaml_color5Path /home/zhouxian/catkin_ws/newData2/colorData/Cam5/data.yaml --yaml_color6Path /home/zhouxian/catkin_ws/newData2/colorData/Cam6/data.yaml --yaml_depth1Path /home/zhouxian/catkin_ws/newData2/depthData/Cam1/data.yaml --yaml_depth2Path /home/zhouxian/catkin_ws/newData2/depthData/Cam2/data.yaml --yaml_depth3Path /home/zhouxian/catkin_ws/newData2/depthData/Cam3/data.yaml --yaml_depth4Path /home/zhouxian/catkin_ws/newData2/depthData/Cam4/data.yaml --yaml_depth5Path /home/zhouxian/catkin_ws/newData2/depthData/Cam5/data.yaml --yaml_depth6Path /home/zhouxian/catkin_ws/newData2/depthData/Cam6/data.yaml
'''

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--yaml_color1Path', type=str, required=True)
    parser.add_argument('--yaml_color2Path', type=str, required=True)
    parser.add_argument('--yaml_color3Path', type=str, required=True)
    parser.add_argument('--yaml_color4Path', type=str, required=True)
    parser.add_argument('--yaml_color5Path', type=str, required=True)
    parser.add_argument('--yaml_color6Path', type=str, required=True)

    parser.add_argument('--yaml_depth1Path', type=str, required=True)
    parser.add_argument('--yaml_depth2Path', type=str, required=True)
    parser.add_argument('--yaml_depth3Path', type=str, required=True)
    parser.add_argument('--yaml_depth4Path', type=str, required=True)
    parser.add_argument('--yaml_depth5Path', type=str, required=True)
    parser.add_argument('--yaml_depth6Path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)

    args=parser.parse_args()
    if not os.path.exists(args.yaml_color1Path):
        print("Invalid Path for color Image File 1")
        os.sys.exit(-1)

    if not os.path.exists(args.yaml_color2Path):
        print("Invalid Path for color Image File 2")
        os.sys.exit(-1)

    if not os.path.exists(args.yaml_color3Path):
        print("Invalid Path for color Image File 3")
        os.sys.exit(-1)

    if not os.path.exists(args.yaml_color4Path):
        print("Invalid Path for color Image File 4")
        os.sys.exit(-1)

    if not os.path.exists(args.yaml_color5Path):
        print("Invalid Path for color Image File 5")
        os.sys.exit(-1)

    if not os.path.exists(args.yaml_color6Path):
        print("Invalid Path for color Image File 6")
        os.sys.exit(-1)

    if not os.path.exists(args.yaml_depth1Path):
        print("Invalid Path for depth Image File 1")
        os.sys.exit(-1)

    if not os.path.exists(args.yaml_depth2Path):
        print("Invalid Path for depth Image File 2")
        os.sys.exit(-1)

    if not os.path.exists(args.yaml_depth3Path):
        print("Invalid Path for depth Image File 3")
        os.sys.exit(-1)

    if not os.path.exists(args.yaml_depth4Path):
        print("Invalid Path for depth Image File 4")
        os.sys.exit(-1)

    if not os.path.exists(args.yaml_depth5Path):
        print("Invalid Path for depth Image File 5")
        os.sys.exit(-1)

    if not os.path.exists(args.yaml_depth6Path):
        print("Invalid Path for depth Image File 6")
        os.sys.exit(-1)
    
    with open(args.yaml_color1Path, 'r') as f:
        colorData1=yaml.load(f)

    with open(args.yaml_color2Path, 'r') as f:
        colorData2=yaml.load(f)

    with open(args.yaml_color3Path, 'r') as f:
        colorData3=yaml.load(f)
        #import pdb; pdb.set_trace()

    with open(args.yaml_color4Path, 'r') as f:
        colorData4=yaml.load(f)

    with open(args.yaml_color5Path, 'r') as f:
        colorData5=yaml.load(f)
        #import pdb; pdb.set_trace()

    with open(args.yaml_color6Path, 'r') as f:
        colorData6=yaml.load(f)

    with open(args.yaml_depth1Path, 'r') as f:
        data1=yaml.load(f)

    with open(args.yaml_depth2Path, 'r') as f:
        data2=yaml.load(f)

    with open(args.yaml_depth3Path, 'r') as f:
        data3=yaml.load(f)
        #import pdb; pdb.set_trace()

    with open(args.yaml_depth4Path, 'r') as f:
        data4=yaml.load(f)

    with open(args.yaml_depth5Path, 'r') as f:
        data5=yaml.load(f)
    #import pdb; pdb.set_trace()

    with open(args.yaml_depth6Path, 'r') as f:
        data6=yaml.load(f)
    '''

    with open(args.yaml_color1Path, 'r') as f:
        data1=yaml.load(f)

    with open(args.yaml_color2Path, 'r') as f:
        data2=yaml.load(f)

    with open(args.yaml_color3Path, 'r') as f:
        data3=yaml.load(f)
        #import pdb; pdb.set_trace()

    with open(args.yaml_color4Path, 'r') as f:
        data4=yaml.load(f)

    with open(args.yaml_color5Path, 'r') as f:
        data5=yaml.load(f)
        #import pdb; pdb.set_trace()

    with open(args.yaml_color6Path, 'r') as f:
        data6=yaml.load(f)

    with open(args.yaml_depth1Path, 'r') as f:
        colorData1=yaml.load(f)

    with open(args.yaml_depth2Path, 'r') as f:
        colorData2=yaml.load(f)

    with open(args.yaml_depth3Path, 'r') as f:
        colorData3=yaml.load(f)
        #import pdb; pdb.set_trace()

    with open(args.yaml_depth4Path, 'r') as f:
        colorData4=yaml.load(f)

    with open(args.yaml_depth5Path, 'r') as f:
        colorData5=yaml.load(f)
    #import pdb; pdb.set_trace()

    with open(args.yaml_depth6Path, 'r') as f:
        colorData6=yaml.load(f)
'''
    #import pdb; pdb.set_trace()
    
    indices1=[]
    indices2=[]
    indices3=[]
    indices4=[]
    indices5=[]
    indices6=[]

    indices = [indices1, indices2, indices3, indices4, indices5, indices6]

    colorInd1=[]
    colorInd2=[]
    colorInd3=[]
    colorInd4=[]
    colorInd5=[]
    colorInd6=[]
    colorInds = [colorInd1, colorInd2, colorInd3, colorInd4, colorInd5, colorInd6]

    ind=0

    minDiff=0
    # minDiff3=0
    # minDiff4=0
    # minDiff5=0
    # minDiff6=0
    
    color_no1 = len(colorData1.keys())
    color_no2 = len(colorData2.keys())
    color_no3 = len(colorData3.keys())
    color_no4 = len(colorData4.keys())
    color_no5 = len(colorData5.keys())
    color_no6 = len(colorData6.keys())

    colorVals = [color_no1, color_no2, color_no3, color_no4, color_no5, color_no6]
    colorInd = colorVals.index(min(colorVals))

    depth_no1 = len(data1.keys())
    depth_no2 = len(data2.keys())
    depth_no3 = len(data3.keys())
    depth_no4 = len(data4.keys())
    depth_no5 = len(data5.keys())
    depth_no6 = len(data6.keys())

    depthVals = [depth_no1, depth_no2, depth_no3, depth_no4, depth_no5, depth_no6]
    depthInd = depthVals.index(min(depthVals))
    
    depth_flag = 0
    color_flag = 0

    #comp = [colorVals[colorInd],depthVals[depthInd]]
    if colorVals[colorInd] >= depthVals[depthInd]:
        depth_flag = 1
    else:
        color_flag = 1
    
    depth_data = [data1, data2, data3, data4, data5, data6]
    color_data = [colorData1, colorData2, colorData3, colorData4, colorData5, colorData6]
    #import pdb; pdb.set_trace()
    #COMPARING DEPTH TIMEFRAMES OF OTHER CAMERAS WITH THE CAMERA RECORDING THE LOWEST TIME FRAME TO GET CORRESPONDING TIMEFRAMES.
    if depth_flag == 1:
        for key2 in depth_data[depthInd].keys():
            count1=0
            for i in range(len(depth_data)):
                minDiff=0
                if i==depthInd:
                    indices[i] = range(depthVals[i])
                    continue
                else:
                    for key in depth_data[i].keys():
                        #import pdb; pdb.set_trace()
                        if (depth_data[i][key]['Timestamp'].secs-depth_data[depthInd][key2]['Timestamp'].secs)== 0:
                            #print("In Here.")
                            count1+=1
                            if count1==1:
                                minDiff = abs(depth_data[i][key]['Timestamp'].nsecs-depth_data[depthInd][key2]['Timestamp'].nsecs)
                                ind=key
                            else:
                                diff=abs(depth_data[i][key]['Timestamp'].nsecs-depth_data[depthInd][key2]['Timestamp'].nsecs)
                                if diff<minDiff:
                                    minDiff=diff
                                    ind=key

                    indices[i].append(ind)
        #import pdb; pdb.set_trace()

        #COMPARING DEPTH TIMESTAMPS OF EACH CAMERA WITH ITS CORRESPONDING COLOR TIMESTAMPS TO GET SYNCED COLOR TIMESTAMPS.
        for i in range(len(indices)):
            #import pdb; pdb.set_trace()
            # print("-----------------------------------------------------------")
            # print(i)
            ind=0
            minDiff1=0
            for j in range(len(indices[i])):
                count1=0
                for key in color_data[i].keys():
                    if (color_data[i][key]['Timestamp'].secs-depth_data[i][j]['Timestamp'].secs)== 0:
                        count1+=1
                        if count1==1:
                            minDiff1 = abs(color_data[i][key]['Timestamp'].nsecs-depth_data[i][j]['Timestamp'].nsecs)
                            ind=key
                        else:
                            diff=abs(color_data[i][key]['Timestamp'].nsecs-depth_data[i][j]['Timestamp'].nsecs)
                            if diff<minDiff1:
                                minDiff1=diff
                                ind=key
                colorInds[i].append(ind)
            #pdb.set_trace()

    else:
        for key2 in color_data[colorInd].keys():
            count1=0
            for i in range(len(color_data)):
                minDiff=0
                if i==colorInd:
                    colorInds[i] = range(colorVals[i])
                    continue
                else:
                    for key in color_data[i].keys():
                        if (color_data[i][key]['Timestamp'].secs-color_data[colorInd][key2]['Timestamp'].secs)== 0:
                            count1+=1
                            if count1==1:
                                minDiff = abs(color_data[i][key]['Timestamp'].nsecs-color_data[colorInd][key2]['Timestamp'].nsecs)
                                ind=key
                            else:
                                diff=abs(color_data[i][key]['Timestamp'].nsecs-color_data[colorInd][key2]['Timestamp'].nsecs)
                                if diff<minDiff:
                                    minDiff=diff
                                    ind=key

                    colorInds[i].append(ind)

        #COMPARING COLOR TIMESTAMPS OF EACH CAMERA WITH ITS CORRESPONDING DEPTH TIMESTAMPS TO GET SYNCED DEPTH TIMESTAMPS.
        for i in range(len(colorInds)):
            ind=0
            minDiff1=0
            for j in range(len(colorInds[i])):
                count1=0
                for key in depth_data[i].keys():
                    #import pdb; pdb.set_trace()
                    #print(depth_data[i][key]['Timestamp'].secs)
                    #print(color_data[i][j]['Timestamp'].secs)
                    if (depth_data[i][key]['Timestamp'].secs-color_data[i][j]['Timestamp'].secs)== 0:
                        count1+=1
                        if count1==1:
                            minDiff1 = abs(depth_data[i][key]['Timestamp'].nsecs-color_data[i][j]['Timestamp'].nsecs)
                            ind=key
                        else:
                            diff=abs(depth_data[i][key]['Timestamp'].nsecs-color_data[i][j]['Timestamp'].nsecs)
                            if diff<minDiff1:
                                minDiff1=diff
                                ind=key
                indices[i].append(ind)
    

    indexData={"colorIndex1":colorInds[0], "colorIndex2":colorInds[1], "colorIndex3":colorInds[2], "colorIndex4":colorInds[3], "colorIndex5":colorInds[4], "colorIndex6":colorInds[5], "depthIndex1":indices[0], "depthIndex2":indices[1], "depthIndex3": indices[2], "depthIndex4": indices[3], "depthIndex5": indices[4], "depthIndex6": indices[5]}
    '''
    
    indexData={"colorIndex1": indices1, "colorIndex2": indices2, "colorIndex3": indices3, "colorIndex4": indices4, "colorIndex5": range(687), "colorIndex6": indices6, "depthIndex1":colorInd1, "depthIndex2":colorInd2, "depthIndex3": colorInd3, "depthIndex4": colorInd4, "depthIndex5": colorInd5, "depthIndex6": colorInd6}
    '''
    print("save", os.path.join(args.save_dir, "syncedIndexData.pkl"))
    #import pdb;pdb.set_trace()
    pickle.dump(indexData, open(os.path.join(args.save_dir, "syncedIndexData.pkl"), "wb"))



if __name__ =="__main__":


    main()