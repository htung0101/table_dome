import numpy as np
import argparse
import pdb
import os
import pickle
import PIL.Image as Image

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--colorImage1Path', type=str, required=True)
    parser.add_argument('--colorImage2Path', type=str, required=True)
    parser.add_argument('--colorImage3Path', type=str, required=True)
    parser.add_argument('--colorImage4Path', type=str, required=True)
    parser.add_argument('--colorImage5Path', type=str, required=True)
    parser.add_argument('--colorImage6Path', type=str, required=True)

    parser.add_argument('--depthImage1Path', type=str, required=True)
    parser.add_argument('--depthImage2Path', type=str, required=True)
    parser.add_argument('--depthImage3Path', type=str, required=True)
    parser.add_argument('--depthImage4Path', type=str, required=True)
    parser.add_argument('--depthImage5Path', type=str, required=True)
    parser.add_argument('--depthImage6Path', type=str, required=True)

    args=parser.parse_args()

    if not os.path.exists(args.colorImage1Path):
        print("Camera 1 color image path doesnot exist.")
        os.sys.exit(-1)

    if not os.path.exists(args.colorImage2Path):
        print("Camera 2 color image path doesnot exist.")
        os.sys.exit(-1)

    if not os.path.exists(args.colorImage3Path):
        print("Camera 3 color image path doesnot exist.")
        os.sys.exit(-1)

    if not os.path.exists(args.colorImage4Path):
        print("Camera 4 color image path doesnot exist.")
        os.sys.exit(-1)

    if not os.path.exists(args.colorImage5Path):
        print("Camera 5 color image path doesnot exist.")
        os.sys.exit(-1)

    if not os.path.exists(args.colorImage6Path):
        print("Camera 6 color image path doesnot exist.")
        os.sys.exit(-1)

    if not os.path.exists(args.depthImage1Path):
        print("Camera 1 depth image path doesnot exist.")
        os.sys.exit(-1)

    if not os.path.exists(args.depthImage2Path):
        print( "Camera 2 depth image path doesnot exist.")
        os.sys.exit(-1)

    if not os.path.exists(args.depthImage3Path):
        print("Camera 3 depth image path doesnot exist.")
        os.sys.exit(-1)

    if not os.path.exists(args.depthImage4Path):
        print("Camera 4 depth image path doesnot exist.")
        os.sys.exit(-1)

    if not os.path.exists(args.depthImage5Path):
        print("Camera 5 depth image path doesnot exist.")
        os.sys.exit(-1)

    if not os.path.exists(args.depthImage6Path):
        print("Camera 6 depth image path doesnot exist.")
        os.sys.exit(-1)

    img1 = Image.open(args.colorImage1Path)
    img1 = np.asarray(img1)

    img2 = np.asarray(Image.open(args.colorImage2Path))
    img3 = np.asarray(Image.open(args.colorImage3Path))
    img4 = np.asarray(Image.open(args.colorImage4Path))
    img5 = np.asarray(Image.open(args.colorImage5Path))
    img6 = np.asarray(Image.open(args.colorImage6Path))

    depth_img1 = np.asarray(Image.open(args.depthImage1Path))
    depth_img2 = np.asarray(Image.open(args.depthImage2Path))
    depth_img3 = np.asarray(Image.open(args.depthImage3Path))
    depth_img4 = np.asarray(Image.open(args.depthImage4Path))
    depth_img5 = np.asarray(Image.open(args.depthImage5Path))
    depth_img6 = np.asarray(Image.open(args.depthImage6Path))

    save_dict = {
        'img1': img1, 'img2': img2, 'img3':img3,
        'img4': img4, 'img5': img5, 'img6': img6,

        'depth_img1': depth_img1, 'depth_img2': depth_img2,
        'depth_img3': depth_img3, 'depth_img4': depth_img4,
        'depth_img5': depth_img5, 'depth_img6': depth_img6

    }

    with open('all_images.pkl', 'wb') as f:
        pickle.dump(save_dict, f)

    f.close()

    from IPython import embed; embed()

if __name__ == '__main__':
    main()

