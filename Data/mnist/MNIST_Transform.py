import struct
from array import array
import os
# 通过 pip install pypng 命令安装此库
import png

trainimg = './train-images.idx3-ubyte'
trainlabel = './train-labels.idx1-ubyte'
testimg = './t10k-images.idx3-ubyte'
testlabel = './t10k-labels.idx1-ubyte'
trainfolder = './train'
testfolder = './test'
if not os.path.exists(trainfolder): os.makedirs(trainfolder)
if not os.path.exists(testfolder): os.makedirs(testfolder)

# open(文件路径，读写格式)，用于打开一个文件，返回一个文件对象
# rb表示以二进制读模式打开文件
trimg = open(trainimg, 'rb')
teimg = open(testimg, 'rb')
trlab = open(trainlabel, 'rb')
telab = open(testlabel, 'rb')
# struct的用法这里不详述
struct.unpack(">IIII", trimg.read(16))
struct.unpack(">IIII", teimg.read(16))
struct.unpack(">II", trlab.read(8))
struct.unpack(">II", telab.read(8))
# array模块是Python中实现的一种高效的数组存储类型
# 所有数组成员都必须是同一种类型，在创建数组时就已经规定
# B表示无符号字节型，b表示有符号字节型
trimage = array("B", trimg.read())
teimage = array("B", teimg.read())
trlabel = array("b", trlab.read())
telabel = array("b", telab.read())
# close方法用于关闭一个已打开的文件，关闭后文件不能再进行读写操作
trimg.close()
teimg.close()
trlab.close()
telab.close()
# 为训练集和测试集各定义10个子文件夹，用于存放从0到9的所有数字，文件夹名分别为0-9
trainfolders = [os.path.join(trainfolder, str(i)) for i in range(10)]
testfolders = [os.path.join(testfolder, str(i)) for i in range(10)]
for dir in trainfolders:
    if not os.path.exists(dir):
        os.makedirs(dir)
for dir in testfolders:
    if not os.path.exists(dir):
        os.makedirs(dir)
# 开始保存训练图像数据
for (i, label) in enumerate(trlabel):
    filename = os.path.join(trainfolders[label], str(i) + ".png")
    print("writing " + filename)
    with open(filename, "wb") as img:
        image = png.Writer(28, 28, greyscale=True)
        data = [trimage[(i * 28 * 28 + j * 28): (i * 28 * 28 + (j + 1) * 28)] for j in range(28)]
        image.write(img, data)
# 开始保存测试图像数据
for (i, label) in enumerate(telabel):
    filename = os.path.join(testfolders[label], str(i) + ".png")
    print("writing " + filename)
    with open(filename, "wb") as img:
        image = png.Writer(28, 28, greyscale=True)
        data = [teimage[(i * 28 * 28 + j * 28): (i * 28 * 28 + (j + 1) * 28)] for j in range(28)]
        image.write(img, data)