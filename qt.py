import queue
import sys

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from keras.models import load_model

import picture as pic


class Ui_MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)  # 父类的构造函数

        self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.cap = cv2.VideoCapture()  # 视频流
        self.CAM_NUM = 0  # 为0时表示视频流来自笔记本内置摄像头

        self.set_ui()  # 初始化程序界面
        self.slot_init()  # 初始化槽函数
        self.frame_num = 0
        self.frames = []
        self.inference_frames = []
        self.action_labels = ['step1', 'step2', 'step3', 'step4', 'step5', 'step6', 'step7']
        self.inference_model = load_model('myModel.h5')
        self.complete = {}

        self.x0 = 300
        self.y0 = 100
        self.width = 300
        self.height = 300
        self.actionQueue = queue.Queue()

        for step in self.action_labels:
            self.actionQueue.put(step)
            self.complete[step] = 5

        self.current_stage = 'step1'

    '''程序界面布局'''

    def set_ui(self):
        self.__layout_main = QtWidgets.QHBoxLayout()  # 总布局
        self.__layout_fun_button = QtWidgets.QVBoxLayout()  # 按键布局
        self.__layout_data_show = QtWidgets.QVBoxLayout()  # 数据(视频)显示布局
        self.__layout_roi_res_show = QtWidgets.QVBoxLayout()  # 数据(视频)显示布局
        self.button_open_camera = QtWidgets.QPushButton('Open Camera')  # 建立用于打开摄像头的按键
        self.button_close = QtWidgets.QPushButton('Exit')  # 建立用于退出程序的按键
        self.button_open_camera.setMinimumHeight(50)  # 设置按键大小
        self.button_close.setMinimumHeight(50)
        self.button_close.move(10, 100)  # 移动按键

        '''信息显示'''
        self.label_show_camera = QtWidgets.QLabel()  # 定义显示视频的Label
        self.label_show_camera.setFixedSize(641, 481)  # 给显示视频的Label设置大小为641x481
        self.__layout_data_show.addWidget(self.label_show_camera)

        self.label_roi_camera = QtWidgets.QLabel()  # 定义显示视频roi的Label
        self.label_roi_camera.setFixedSize(201, 241)  # 给显示视频roi的Label设置大小为300x240
        self.__layout_roi_res_show.addWidget(self.label_roi_camera)

        self.label_res_camera = QtWidgets.QLabel()  # 定义显示视频res的Label
        self.label_res_camera.setFixedSize(201, 241)  # 给显示视频res的Label设置大小为300x240
        self.__layout_roi_res_show.addWidget(self.label_res_camera)

        '''把按键加入到按键布局中'''
        self.__layout_fun_button.addWidget(self.button_open_camera)  # 把打开摄像头的按键放到按键布局中
        self.__layout_fun_button.addWidget(self.button_close)  # 把退出程序的按键放到按键布局中
        '''把某些控件加入到总布局中'''
        self.__layout_main.addLayout(self.__layout_fun_button)  # 把按键布局加入到总布局中
        self.__layout_main.addLayout(self.__layout_data_show)  # 把用于显示视频的Label加入到总布局中
        self.__layout_main.addLayout(self.__layout_roi_res_show)
        '''总布局布置好后就可以把总布局作为参数传入下面函数'''
        self.setLayout(self.__layout_main)  # 到这步才会显示所有控件

    '''初始化所有槽函数'''

    def slot_init(self):
        self.button_open_camera.clicked.connect(
            self.button_open_camera_clicked)  # 若该按键被点击，则调用button_open_camera_clicked()
        self.timer_camera.timeout.connect(self.show_camera)  # 若定时器结束，则调用show_camera()
        self.button_close.clicked.connect(self.close)  # 若该按键被点击，则调用close()，注意这个close是父类QtWidgets.QWidget自带的，会关闭程序

    '''槽函数之一'''

    def button_open_camera_clicked(self):
        if not self.timer_camera.isActive():  # 若定时器未启动
            flag = self.cap.open(self.CAM_NUM)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if not flag:  # flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "pleas check your camera setting on your laptop",
                                                    buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.button_open_camera.setText('Close Camera')
        else:
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.label_show_camera.clear()  # 清空视频显示区域
            self.label_res_camera.clear()
            self.label_roi_camera.clear()
            self.button_open_camera.setText('Open Camera')

    def show_camera(self):
        flag, image = self.cap.read()  # 从视频流中读取
        image = cv2.flip(image, 2)

        text1 = "Please start wish your hand"
        text2 = "You are in {} now ({})".format(self.current_stage, self.complete[self.current_stage])

        roi, res = pic.binaryMask(image, self.x0, self.y0, self.width, self.height)  # 取手势所在框图并进行处理

        show = cv2.resize(image, (640, 480))  # 把读到的帧的大小重新设置为 640x480
        roi = cv2.resize(roi, (300, 240))
        res = cv2.resize(res, (300, 240))

        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

        if self.frame_num == 10:

            inference_video = np.array(self.frames)
            inference_video = np.array([inference_video]).transpose((0, 2, 3, 1))
            inference_video = np.array([inference_video]).reshape((inference_video.shape[0], 32, 32, 10, 1))

            prdict = self.inference_model.predict(inference_video)[0]
            proba = prdict[np.argmax(prdict)]

            result = self.action_labels[np.argmax(prdict)]

            threshold = 0.7
            if proba < threshold:
                result = None

            print("this is {} and the probability is {}".format(result, proba))

            self.frame_num = 0
            self.frames = []

            if not result is None:
                if result == self.current_stage:
                    self.complete[result] -= 1
                if self.complete[result] == 0:
                    self.current_stage = self.actionQueue.get()
                    if self.actionQueue.empty:
                        text1 = "You have finished"
                        text2 = " "

        else:
            self.frames.append(cv2.resize(cv2.cvtColor(res, cv2.COLOR_BGR2GRAY), (32, 32)))
            self.frame_num += 1

        show = cv2.putText(show, text1, (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        show = cv2.putText(show, text2, (40, 65), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)

        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                 QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式

        showRoi = QtGui.QImage(roi.data, roi.shape[1], roi.shape[0],
                               QtGui.QImage.Format_RGB888)
        showRes = QtGui.QImage(res.data, res.shape[1], res.shape[0],
                               QtGui.QImage.Format_RGB888)

        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage
        self.label_roi_camera.setPixmap(QtGui.QPixmap.fromImage(showRoi))
        self.label_res_camera.setPixmap(QtGui.QPixmap.fromImage(showRes))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)  # 固定的，表示程序应用
    ui = Ui_MainWindow()  # 实例化Ui_MainWindow
    ui.show()  # 调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的
    sys.exit(app.exec_())
