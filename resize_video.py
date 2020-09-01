import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input video file")
ap.add_argument("-o", "--output", required=True,
        help="Video output of facelandmark")
ap.add_argument("-c", "--codec", type=str, default="MJPG",
        help="codec of output video")
args = vars(ap.parse_args())
# Capture video
cap = cv2.VideoCapture(args["input"])
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
video_opt_path = args["output"]
out = cv2.VideoWriter(video_opt_path, cv2.VideoWriter_fourcc(*args["codec"]), 20, (1920//2, 1080//2), True)
i = 0
while True:
    ret, img = cap.read()
    img = cv2.resize(img, (1920//2, 1080//2), interpolation = cv2.INTER_LINEAR)
    # print(img.shape)
    if img is not None:
        out.write(img)
        cv2.imshow("Resize image", img)
    # if the `q` key was pressed, break from the loop
    # if esc key was pressed, escape video streaming
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key==27:
        cv2.destroyAllWindows()
    i+=1