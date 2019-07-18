import numpy as np
import yaml
import argparse
import os
import pickle

'''
    COMMAND FOR RUNNING THE CODE:
    
    python timeCompare.py --yaml_color1Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/colorData/Cam1/data.yaml --yaml_color2Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/colorData/Cam2/data.yaml --yaml_color3Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/colorData/Cam3/data.yaml --yaml_color4Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/colorData/Cam4/data.yaml --yaml_color5Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/colorData/Cam5/data.yaml --yaml_color6Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/colorData/Cam6/data.yaml --yaml_depth1Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/depthData/Cam1/data.yaml --yaml_depth2Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/depthData/Cam2/data.yaml --yaml_depth3Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/depthData/Cam3/data.yaml --yaml_depth4Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/depthData/Cam4/data.yaml --yaml_depth5Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/depthData/Cam5/data.yaml --yaml_depth6Path /home/zhouxian/catkin_ws/src/calibrate/src/scripts/Data/depthData/Cam6/data.yaml

    python timeCompare.py --yaml_color1Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData5/colorData/Cam1/data.yaml --yaml_color2Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData5/colorData/Cam2/data.yaml --yaml_color3Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData5/colorData/Cam3/data.yaml --yaml_color4Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData5/colorData/Cam4/data.yaml --yaml_color5Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData5/colorData/Cam5/data.yaml --yaml_color6Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData5/colorData/Cam6/data.yaml --yaml_depth1Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData5/depthData/Cam1/data.yaml --yaml_depth2Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData5/depthData/Cam2/data.yaml --yaml_depth3Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData5/depthData/Cam3/data.yaml --yaml_depth4Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData5/depthData/Cam4/data.yaml --yaml_depth5Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData5/depthData/Cam5/data.yaml --yaml_depth6Path /media/zhouxian/Elements/Kalibrbackup/TrainData/newData5/depthData/Cam6/data.yaml

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
    indices6=[]
    indices3=[]
    indices4=[]
    indices1=[]
    indices2=[]
    ind=0
    minDiff1=0
    minDiff3=0
    minDiff4=0
    minDiff5=0
    minDiff6=0

    #COMPARING DEPTH TIMEFRAMES OF OTHER CAMERAS WITH THE CAMERA RECORDING THE LOWEST TIME FRAME TO GET CORRESPONDING TIMEFRAMES.
    for key2 in data5.keys():
        count1=0
        count3=0
        count4=0
        count5=0
        count6=0
        for key in data6.keys():
            if (data6[key]['Timestamp'].secs-data5[key2]['Timestamp'].secs)== 0:
                count1+=1
                if count1==1:
                    minDiff1 = abs(data6[key]['Timestamp'].nsecs-data5[key2]['Timestamp'].nsecs)
                    ind=key
                else:
                    diff=abs(data6[key]['Timestamp'].nsecs-data5[key2]['Timestamp'].nsecs)
                    if diff<minDiff1:
                        minDiff1=diff
                        ind=key

        indices6.append(ind)

        for key in data3.keys():
            if (data3[key]['Timestamp'].secs-data5[key2]['Timestamp'].secs)== 0:
                count3+=1
                if count3==1:
                    minDiff3 = abs(data3[key]['Timestamp'].nsecs-data5[key2]['Timestamp'].nsecs)
                    ind=key
                else:
                    diff=abs(data3[key]['Timestamp'].nsecs-data5[key2]['Timestamp'].nsecs)
                    if diff<minDiff3:
                        minDiff3=diff
                        ind=key
        indices3.append(ind)

        for key in data4.keys():
            if (data4[key]['Timestamp'].secs-data5[key2]['Timestamp'].secs)== 0:
                count4+=1
                if count4==1:
                    minDiff4 = abs(data4[key]['Timestamp'].nsecs-data5[key2]['Timestamp'].nsecs)
                    ind=key
                else:
                    diff=abs(data4[key]['Timestamp'].nsecs-data5[key2]['Timestamp'].nsecs)
                    if diff<minDiff4:
                        minDiff4=diff
                        ind=key
        indices4.append(ind)
        #import pdb;pdb.set_trace()
        for key in data1.keys():
            print("data1 Keys:",key)
            if (data1[key]['Timestamp'].secs-data5[key2]['Timestamp'].secs)== 0:
                count5+=1
                if count5==1:
                    minDiff5 = abs(data1[key]['Timestamp'].nsecs-data5[key2]['Timestamp'].nsecs)
                    ind=key
                else:
                    diff=abs(data1[key]['Timestamp'].nsecs-data5[key2]['Timestamp'].nsecs)
                    if diff<minDiff5:
                        minDiff5=diff
                        ind=key
        indices1.append(ind)
        #import pdb;pdb.set_trace()
        for key in data2.keys():
            if (data2[key]['Timestamp'].secs-data5[key2]['Timestamp'].secs)== 0:
                count6+=1
                if count6==1:
                    minDiff6 = abs(data2[key]['Timestamp'].nsecs-data5[key2]['Timestamp'].nsecs)
                    ind=key
                else:
                    diff=abs(data2[key]['Timestamp'].nsecs-data5[key2]['Timestamp'].nsecs)
                    if diff<minDiff6:
                        minDiff6=diff
                        ind=key
        indices2.append(ind)

    #COMPARING DEPTH TIMESTAMPS OF EACH CAMERA WITH ITS CORRESPONDING COLOR TIMESTAMPS TO GET SYNCED COLOR TIMESTAMPS.

    ind=0
    # minDiff1=0
    # minDiff3=0
    # minDiff4=0
    # minDiff5=0
    # minDiff6=0
    colorInd1=[]
    colorInd2=[]
    colorInd3=[]
    colorInd4=[]
    colorInd5=[]
    colorInd6=[]

    for i in indices1:
        count1=0
        for key in colorData1.keys():
            if (colorData1[key]['Timestamp'].secs-data1[i]['Timestamp'].secs)== 0:
                count1+=1
                if count1==1:
                    minDiff1 = abs(colorData1[key]['Timestamp'].nsecs-data1[i]['Timestamp'].nsecs)
                    ind=key
                else:
                    diff=abs(colorData1[key]['Timestamp'].nsecs-data1[i]['Timestamp'].nsecs)
                    if diff<minDiff1:
                        minDiff1=diff
                        ind=key
        colorInd1.append(ind)

    for i in indices2:
        count2=0
        for key in colorData2.keys():
            if (colorData2[key]['Timestamp'].secs-data2[i]['Timestamp'].secs)== 0:
                count2+=1
                if count2==1:
                    minDiff2 = abs(colorData2[key]['Timestamp'].nsecs-data2[i]['Timestamp'].nsecs)
                    ind=key
                else:
                    diff=abs(colorData2[key]['Timestamp'].nsecs-data2[i]['Timestamp'].nsecs)
                    if diff<minDiff2:
                        minDiff2=diff
                        ind=key
        colorInd2.append(ind)

    for i in indices3:
        count3=0
        for key in colorData3.keys():
            if (colorData3[key]['Timestamp'].secs-data3[i]['Timestamp'].secs)== 0:
                count3+=1
                if count3==1:
                    minDiff3 = abs(colorData3[key]['Timestamp'].nsecs-data3[i]['Timestamp'].nsecs)
                    ind=key
                else:
                    diff=abs(colorData3[key]['Timestamp'].nsecs-data3[i]['Timestamp'].nsecs)
                    if diff<minDiff3:
                        minDiff3=diff
                        ind=key
        colorInd3.append(ind)

    for i in indices4:
        count4=0
        for key in colorData4.keys():
            if (colorData4[key]['Timestamp'].secs-data4[i]['Timestamp'].secs)== 0:
                count4+=1
                if count4==1:
                    minDiff4 = abs(colorData4[key]['Timestamp'].nsecs-data4[i]['Timestamp'].nsecs)
                    ind=key
                else:
                    diff=abs(colorData4[key]['Timestamp'].nsecs-data4[i]['Timestamp'].nsecs)
                    if diff<minDiff4:
                        minDiff4=diff
                        ind=key
        colorInd4.append(ind)

    for i in range(430):
        count5=0
        for key in colorData5.keys():
            if (colorData5[key]['Timestamp'].secs-data5[i]['Timestamp'].secs)== 0:
                count5+=1
                if count5==1:
                    minDiff5 = abs(colorData5[key]['Timestamp'].nsecs-data5[i]['Timestamp'].nsecs)
                    ind=key
                else:
                    diff=abs(colorData5[key]['Timestamp'].nsecs-data5[i]['Timestamp'].nsecs)
                    if diff<minDiff5:
                        minDiff5=diff
                        ind=key
        colorInd5.append(ind)

    for i in indices6:
        count6=0
        for key in colorData6.keys():
            if (colorData6[key]['Timestamp'].secs-data6[i]['Timestamp'].secs)== 0:
                count6+=1
                if count6==1:
                    minDiff6 = abs(colorData6[key]['Timestamp'].nsecs-data6[i]['Timestamp'].nsecs)
                    ind=key
                else:
                    diff=abs(colorData6[key]['Timestamp'].nsecs-data6[i]['Timestamp'].nsecs)
                    if diff<minDiff6:
                        minDiff6=diff
                        ind=key
        colorInd6.append(ind)

    

    indexData={"colorIndex1":colorInd1, "colorIndex2":colorInd2, "colorIndex3":colorInd3, "colorIndex4":colorInd4, "colorIndex5":colorInd5, "colorIndex6":colorInd6, "depthIndex1":indices1, "depthIndex2":indices2, "depthIndex3": indices3, "depthIndex4": indices4, "depthIndex5": range(430), "depthIndex6": indices6}
    '''
    
    indexData={"colorIndex1": indices1, "colorIndex2": indices2, "colorIndex3": indices3, "colorIndex4": indices4, "colorIndex5": range(687), "colorIndex6": indices6, "depthIndex1":colorInd1, "depthIndex2":colorInd2, "depthIndex3": colorInd3, "depthIndex4": colorInd4, "depthIndex5": colorInd5, "depthIndex6": colorInd6}
    '''

    pickle.dump(indexData, open("syncedIndexData.pkl", "wb"))



if __name__ =="__main__":
    main()
