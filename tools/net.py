# net.py is the network architecture of MNIST classification task.
import torch.nn as nn
import torch.nn.functional as F
 
class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)  # 10分类
 
    def forward(self, x):
        in_size = x.size(0)           # BATCH_SIZE=512，输入的x:512*1*28*28。
        out = self.conv1(x)           # batch*1*28*28 -> batch*10*24*24（28x28的图像经过一次核为5x5的卷积，输出变为24x24）
        out = F.relu(out)             # batch*10*24*24
        out = F.max_pool2d(out, 2, 2)  # batch*10*24*24 -> batch*10*12*12（2*2的池化层会减半）
        out = self.conv2(out)         # batch*10*12*12 -> batch*20*10*10（再卷积一次，核的大小是3）
        out = F.relu(out)             # batch*20*10*10
        out = out.view(in_size, -1)   # batch*20*10*10 -> batch*2000（out的第二维是-1，进行自动推算）
        out = self.fc1(out)           # batch*2000 -> batch*500通过
        out = F.relu(out)             # batch*500
        out = self.fc2(out)           # batch*500 -> batch*10
        out = F.log_softmax(out, dim=1)  # 计算log(softmax(x))
        return out