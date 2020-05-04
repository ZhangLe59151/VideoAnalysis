import cv2
import numpy as np


def binaryMask(frame, x0, y0, width, height):
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0))  # 画出截取的手势框图
    roi = frame[y0:y0 + height, x0:x0 + width]  # 获取手势框图
    cv2.imshow("roi", roi)  # 显示手势框图
    res = skinMask(roi)  # 进行肤色检测
    cv2.imshow("res", res)  # 显示肤色检测后的图像
    return res


##########方法一###################
##########BGR空间的手势识别#########
def skinMask(roi):
	skinCrCbHist = np.zeros((256,256), dtype= np.uint8)
	cv2.ellipse(skinCrCbHist, (113,155),(23,25), 43, 0, 360, (255,255,255), -1) #绘制椭圆弧线
	YCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB) #转换至YCrCb空间
	(y,Cr,Cb) = cv2.split(YCrCb) #拆分出Y,Cr,Cb值
	skin = np.zeros(Cr.shape, dtype = np.uint8) #掩膜
	(x,y) = Cr.shape
	for i in range(0, x):
		for j in range(0, y):
			if skinCrCbHist [Cr[i][j], Cb[i][j]] > 0: #若不在椭圆区间中
				skin[i][j] = 255
	res = cv2.bitwise_and(roi,roi, mask = skin)
	return res
