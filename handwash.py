import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import sys
from keras.models import load_model
import queue
import pp as pic

font = cv2.FONT_HERSHEY_SIMPLEX  # 设置字体
size = 0.5  # 设置大小

width, height = 300, 300  # 设置拍摄窗口大小
x0, y0 = 300, 100  # 设置选取位置

SAMPLE_DURATION = 10
SAMPLE_SIZE = 32

ucf_action_labels = ['step1', 'step2', 'step3', 'step4', 'step5', 'step6', 'step7']
testModel = load_model('myModel.h5')
duration = {}

actionQueue = queue.Queue()

for step in ucf_action_labels:
    actionQueue.put(step)
    duration[step] = 5

# test = c3d.load_testdata('test.mp4', c3d.vid3d)
# yy = testModel.predict(x_test)
# proba = testModel.predict(test)[0]

# print("The video is a %s " % (ucf_action_labels[np.argmax(proba)]))

cap = cv2.VideoCapture(0)  # 开摄像头

CURRENT_STAGE = actionQueue.get()

while (1):
    frames = []
    resframes = []

    for i in range(0, SAMPLE_DURATION):

        grabbed, frame = cap.read()  # 读取摄像头的内容
        if not grabbed:
            print("[INFO] no frame read from stream - exqiting")
            sys.exit(0)
        frames.append(frame)

        resframe = cv2.resize(frame[y0:y0 + height, x0:x0 + width], (SAMPLE_SIZE, SAMPLE_SIZE))
        resframe = cv2.cvtColor(resframe, cv2.COLOR_BGR2GRAY)
        resframes.append(resframe)

    video = np.array(resframes)
    video = np.array([video]).transpose((0, 2, 3, 1))
    video = video.reshape((video.shape[0], SAMPLE_SIZE, SAMPLE_SIZE, SAMPLE_DURATION, 1))

    prdict = testModel.predict(video)[0]

    proba = prdict[np.argmax(prdict)]

    result = ucf_action_labels[np.argmax(prdict)]

    print("this is {} and the probability is {}".format(result, proba))

    threshold = 0.9

    if (proba < threshold):
        prdict = "None"

    key = cv2.waitKey(1) & 0xFF  # 按键判断并进行一定的调整
    # 按'j''l''u''j'分别将选框左移，右移，上移，下移
    # 按'q'键退出录像
    if key == ord('i'):
        y0 += 5
    elif key == ord('k'):
        y0 -= 5
    elif key == ord('l'):
        x0 += 5
    elif key == ord('j'):
        x0 -= 5
    if key == ord('q'):
        break

    for i in range(0, SAMPLE_DURATION):
        f = cv2.flip(frames[i], 2)
        roi = pic.binaryMask(f, x0, y0, width, height)
        text1 = 'Please start to wash your hand'
        if result == CURRENT_STAGE:
            duration[CURRENT_STAGE] -= 1
            if duration[CURRENT_STAGE] == 0:
                CURRENT_STAGE = actionQueue.get()
                if actionQueue.empty:
                    cv2.putText(f, 'You have finished', (40, 65), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
                    cv2.imshow('frame', f)
        text2 = 'you have not finishe ' + CURRENT_STAGE + ' yet(' + str(duration[CURRENT_STAGE]) + ')'
        cv2.putText(f, text1, (40, 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        cv2.putText(f, text2, (40, 65), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        cv2.imshow('frame', f)

cap.release()
cv2.destroyAllWindows()  # 关闭所有窗口
