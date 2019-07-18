import cv2
import os

def create_video(image_folder, video_name):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort(key=lambda f: int(filter(str.isdigit, f)))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 60, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def main():
    image_folder = '/home/zhouxian/372019'
    video_folder = '/home/zhouxian/372019_video'
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    num_cameras = 6
    for i in range(1,num_cameras+1):
        img_folder = os.path.join(image_folder, '{}'.format(str(i)))
        video_name = os.path.join(video_folder, "{}.avi".format(i))

        create_video(img_folder, video_name)

if __name__ == '__main__':
    main()
