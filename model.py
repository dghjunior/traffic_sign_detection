# import dependencies

from torch.nn import Module, Conv2d, Linear, MaxPool2d, ReLU, Softmax, Sequential, BatchNorm2d, Sigmoid
from torch import flatten, concat

class HarperNet(Module):
    def __init__(self, numChannels, classes):
        # call the parent constructor
        super(HarperNet, self).__init__()

        # block 1
        self.depth_conv1 = Conv2d(in_channels=numChannels, out_channels=numChannels, kernel_size=3, groups=numChannels)
        self.point_conv1 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.SepConv1 = Sequential(self.depth_conv1, self.point_conv1)
        self.relu1 = ReLU()
        self.depth_conv2 = Conv2d(in_channels=numChannels, out_channels=numChannels, kernel_size=3, groups=numChannels)
        self.point_conv2 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.SepConv2 = Sequential(self.depth_conv2, self.point_conv2)
        self.relu2 = ReLU()
        self.depth_conv3 = Conv2d(in_channels=numChannels, out_channels=numChannels, kernel_size=3, groups=numChannels)
        self.point_conv3 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.SepConv3 = Sequential(self.depth_conv3, self.point_conv3)
        self.batch_norm1 = BatchNorm2d(numChannels)
        self.relu3 = ReLU()
        self.max_pool1 = MaxPool2d(kernel_size=2, stride=2)

        # block 2
        self.depth_conv4 = Conv2d(in_channels=numChannels, out_channels=numChannels, kernel_size=3, groups=numChannels)
        self.point_conv4 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.SepConv4 = Sequential(self.depth_conv4, self.point_conv4)
        self.relu4 = ReLU()
        self.depth_conv5 = Conv2d(in_channels=numChannels, out_channels=numChannels, kernel_size=3, groups=numChannels)
        self.point_conv5 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.SepConv5 = Sequential(self.depth_conv5, self.point_conv5)
        self.relu5 = ReLU()
        self.depth_conv6 = Conv2d(in_channels=numChannels, out_channels=numChannels, kernel_size=3, groups=numChannels)
        self.point_conv6 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.SepConv6 = Sequential(self.depth_conv6, self.point_conv6)
        self.batch_norm2 = BatchNorm2d(numChannels)
        self.relu6 = ReLU()
        self.max_pool2 = MaxPool2d(kernel_size=2, stride=2)

        # block 3
        # inception layer
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.preconv1 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.preconv2 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.max_pool3 = MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.conv3 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.postconv1 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.out1 = concat([self.conv1, self.conv2, self.conv3, self.postconv1])
        # batch_norm, max_pool
        self.batch_norm3 = BatchNorm2d(numChannels)
        self.max_pool6 = MaxPool2d(kernel_size=2, stride=2)

        # block 4
        # inception layer
        self.conv4 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.preconv3 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.preconv4 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.max_pool4 = MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv5 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.conv6 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.postconv2 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.out2 = concat([self.conv4, self.conv5, self.conv6, self.postconv2])
        #  batch_norm, max_pool
        self.batch_norm4 = BatchNorm2d(numChannels)
        self.max_pool7 = MaxPool2d(kernel_size=2, stride=2)

        # block 5
        # inception layer
        self.conv7 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.preconv5 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.preconv6 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.max_pool5 = MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv8 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.conv9 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.postconv3 = Conv2d(in_channels=numChannels, out_channels=32, kernel_size=1)
        self.out3 = concat([self.conv7, self.conv8, self.conv9, self.postconv3])
        #  batch_norm, max_pool
        self.batch_norm5 = BatchNorm2d(numChannels)
        self.max_pool8 = MaxPool2d(kernel_size=2, stride=2)

        # output
        self.sigmoid = Sigmoid()
        self.fc1 = Linear(in_features=32 * 4 * 4, out_features=1)
        self.softmax = Softmax(dim=1)
        self.output = concat([self.sigmoid, self.fc1, self.softmax])

    def forward(self, x):
        # block 1
        x = self.SepConv1(x)
        x = self.relu1(x)
        x = self.SepConv2(x)
        x = self.relu2(x)
        x = self.SepConv3(x)
        x = self.batch_norm1(x)
        x = self.relu3(x)
        x = self.max_pool1(x)

        # block 2
        x = self.SepConv4(x)
        x = self.relu4(x)
        x = self.SepConv5(x)
        x = self.relu5(x)
        x = self.SepConv6(x)
        x = self.batch_norm2(x)
        x = self.relu6(x)
        x = self.max_pool2(x)

        # block 3
        # inception layer
        x1 = self.conv1(x)
        x2 = self.preconv1(x)
        x3 = self.preconv2(x)
        x4 = self.max_pool3(x)
        x4 = self.conv2(x4)
        x5 = self.conv3(x)
        x6 = self.postconv1(x)
        x = concat([x1, x2, x3, x4, x5, x6])
        # batch_norm, max_pool
        x = self.batch_norm3(x)
        x = self.max_pool6(x)

        # block 4
        # inception layer
        x1 = self.conv4(x)
        x2 = self.preconv3(x)
        x3 = self.preconv4(x)
        x4 = self.max_pool4(x)
        x4 = self.conv5(x4)
        x5 = self.conv6(x)
        x6 = self.postconv2(x)
        x = concat([x1, x2, x3, x4, x5, x6])
        # batch_norm, max_pool
        x = self.batch_norm4(x)
        x = self.max_pool7(x)

        # block 5
        # inception layer
        x1 = self.conv7(x)
        x2 = self.preconv5(x)
        x3 = self.preconv6(x)
        x4 = self.max_pool5(x)
        x4 = self.conv8(x4)
        x5 = self.conv9(x)
        x6 = self.postconv3(x)
        x = concat([x1, x2, x3, x4, x5, x6])
        # batch_norm, max_pool
        x = self.batch_norm5(x)
        x = self.max_pool8(x)

        # output
        xcopy = x
        x = self.sigmoid(xcopy)
        y = self.fc1(xcopy)
        z = self.softmax(xcopy)
        x = concat([x, y, z])