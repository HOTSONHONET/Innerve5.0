import numpy as np
from matplotlib import pyplot as plt
import cv2
import io
import time
from FaceDetector.detectFaces import DetectMyFace
import imutils


# Making an instance of our faceDetector Library
faceDetector = DetectMyFace(prototxt_path="FaceDetector//deploy.prototxt", caffeModel_path="FaceDetector//res10_300x300_ssd_iter_140000.caffemodel")

# Making some configuration for video capture
vcap = cv2.VideoCapture(0)
vcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)
vcap.set(cv2.CAP_PROP_FPS, 30)


heartbeat_counts = 128
heartbeat_values = [0]*heartbeat_counts
heartbeat_times = [time.time()]*heartbeat_counts


fig = plt.figure()
ax = fig.add_subplot(111)


while True:
    ret, frame = vcap.read()

    if ret:
        frame = imutils.resize(frame, width=400)
        boxes = faceDetector.detect_faces(frame)
        largest_box = 0
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for box in boxes:
            if box is None:
                continue
            

            startx, starty, endx, endy = box
            face = frame[starty:endy, startx:endx]
            cv2.rectangle(frame, (startx, starty), (endx, endy), (0, 244, 2), 2)
            


            heartbeat_times = heartbeat_times[1:] + [time.time()]
            heartbeat_values = heartbeat_values[1:] + [np.average(face)]


            ax.plot(heartbeat_times, heartbeat_values)
            fig.canvas.draw()
            plot_img_np = np.fromstring(fig.canvas.tostring_rgb(), dtype = np.uint8, sep = '')
            plot_img_np = plot_img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.cla()

            cv2.imshow('CROP', frame)
            cv2.imshow('GRAPH', plot_img_np)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    else:
        print("NO FRAME TO CAPTURE")

vcap.release()
cv2.destroyAllWindows()