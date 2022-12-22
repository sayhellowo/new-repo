# 此代码使用LSTM网络实现对周期序列的预测
import torch

from LSTM import *
from LSTMTagger import *
import torch.optim as optim
import torch.utils.data as Data
import os
import matplotlib.pyplot as plt
import numpy as np


def sin_wave(A, fs, t, phi):
    '''
    :params A:    振幅
    :params fs:   采样频率
    :params phi:  相位
    :params t:    时间长度
    '''
    # 若时间序列长度为 t=1s,
    # 采样频率 fs=1000 Hz, 则采样时间间隔 Ts=1/fs=0.001s
    # 对于时间序列采样点个数为 n=t/Ts=1/0.001=1000, 即有1000个点,每个点间隔为 Ts
    n = np.linspace(0, t, t*fs)
    y = A*np.sin((phi + np.pi * 2) * n)+A

    return np.array(y)

def plot_picture(n, y):
    """
    :param n: 图像的x轴序列
    :param y: 图像值
    :return: None 画出图像
    """
    plt.xlabel('t/s')
    plt.ylabel('y')
    plt.grid()
    plt.plot(n, y, 'k')
    plt.show()


def prepare_data(data, window_size, predict_size, device):
    data_in = []
    data_out = []
    for i in range(data.shape[0] - window_size - predict_size):
        data_in.append(data[i:i + window_size].reshape(1, window_size)[0])
        data_out.append(data[i + window_size:i + window_size + predict_size].reshape(1, predict_size)[0])
    data_train = torch.tensor(np.array(data_in).reshape(-1, window_size), dtype=torch.float).to(device)
    data_predict = torch.tensor(np.array(data_out).reshape(-1, predict_size), dtype=torch.float).to(device)
    # data_train = torch.tensor(np.array(data_in).transpose(), dtype=torch.float).to(device)
    # data_predict = torch.tensor(np.array(data_out).transpose(), dtype=torch.float).to(device)
    return data_train, data_predict


if __name__ == "__main__":
    # 选择GPU或者CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lr = 0.001
    train_epoch = 100
    window_size = 30
    predict_size = 7
    data = sin_wave(A=10, fs=50, t=10, phi=0)
    data_train, data_predict = prepare_data(data, window_size, predict_size, device)


    # 实际中通常使用更大的维度如32维, 64维.
    # 这里我们使用小的维度, 为了方便查看训练过程中权重的变化.
    HIDDEN_DIM = 12

    model = LSTM(window_size, HIDDEN_DIM, predict_size, device)
    # model = LSTMTagger(20, HIDDEN_DIM, window_size, predict_size, device)
    model.to(device)
    print(model)

    # 判断模型在gpu还是cpu上
    print(next(model.parameters()).device)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(train_epoch):  # 实际情况下你不会训练300个周期, 此例中我们只是随便设了一个值
        print(epoch)
        for step, input in enumerate(data_train):
            # 第一步: 请记住Pytorch会累加梯度.
            # 我们需要在训练每个实例前清空梯度
            model.zero_grad()
            # input = torch.tensor(input, dtype=torch.long).to(device)
            # torch.tensor(input).to(device)
            # aa = data_predict[step, :]

            # 此外还需要清空 LSTM 的隐状态,
            # 将其从上个实例的历史中分离出来.
            model.hidden = model.init_hidden()

            # 第三步: 前向传播.
            tag_scores = model(input)
            print(tag_scores)

            # 第四步: 计算损失和梯度值, 通过调用 optimizer.step() 来更新梯度
            # real = torch.tensor(data_predict[step, :], dtype=torch.long).to(device)
            loss = loss_function(tag_scores, data_predict[step, :])
            loss.backward()
            optimizer.step()

    # 查看训练后的得分
    with torch.no_grad():
        inputs = torch.tensor(np.array(data[-1-window_size:]).reshape(-1, window_size), dtype=torch.float).to(device)
        # inputs = torch.tensor(np.array(data[-1-window_size:]).reshape(-1, window_size), dtype=torch.long).to(device)
        tag_scores = model(inputs)

        # 句子是 "the dog ate the apple", i,j 表示对于单词 i, 标签 j 的得分.
        # 我们采用得分最高的标签作为预测的标签. 从下面的输出我们可以看到, 预测得
        # 到的结果是0 1 2 0 1. 因为 索引是从0开始的, 因此第一个值0表示第一行的
        # 最大值, 第二个值1表示第二行的最大值, 以此类推. 所以最后的结果是 DET
        # NOUN VERB DET NOUN, 整个序列都是正确的!
        print(tag_scores)