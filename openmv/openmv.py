# 该部分在OpenMV的查找色块的算法中，运用的就是这个LAB模式！
# 一个颜色阈值的结构是这样的：red = (minL, maxL, minA, maxA, minB, maxB)

import sensor, image, time #引入图像处理和时间模块


# 设置摄像头
sensor.reset()#初始化感光元件
sensor.set_pixformat(sensor.RGB565)#设置为彩色
sensor.set_framesize(sensor.QVGA)#设置图像的大小
sensor.skip_frames()#跳过n张照片，在更改设置后，跳过一些帧，等待感光元件变稳定。
sensor.set_auto_whitebal(False) #关闭自动白平衡


# 一直拍照
while(True):
    img = sensor.snapshot()#拍摄一张照片，img为一个image对象