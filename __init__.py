#coding=utf-8

import numpy as np
from lstm_coling import lstm_31train_31test
import random

if __name__ == '__main__':
    print("lstm is running...")

    #调参结果存储的文件名
    filename_parameter_adjust_result = u"lstm_full_result_not_biaozhun.csv"
    writer = open(filename_parameter_adjust_result, 'w')
    first_line = 'index,layer_num,cell_num,hidden_num,nb_epoch,dropout,lr,batch_size,accuracy'
    writer.write(first_line + '\n')
    writer.close()

    #训练集和测试集的文件名的后一部分
    path = u"./"
    filename_x_test = 'sw_test_x.npy'
    filename_x_train = 'sw_train_x.npy'
    filename_y_test = 'sw_test_y.npy'
    filename_y_train = 'sw_train_y.npy'

    # 参数列表
    layer_nums = [2]  # LSTM层数
    cell_nums = [50, 150, 300]  # 每层LSTM中cell个数
    nb_epochs = [10000]  # 迭代次数
    dropouts = [0.1, 0.3, 0.5]  # 随机因子
    lrs = [0.001]  # 学习率
    batch_sizes = [25]  # 训练批量的大小
    index = int(random.random()*10000) # 模型的标识号

    #参数列表
    parameter_lstm_list = list()
    for nb_epoch in nb_epochs:
        for layer_num in layer_nums:
            for cell_num in cell_nums:
                for dropout in dropouts:
                    for lr in lrs:
                        for batch_size in batch_sizes:

                            param = dict()
                            param['layer_num'] = layer_num
                            param['cell_num'] = cell_num
                            param['nb_epoch'] = nb_epoch
                            param['dropout'] = dropout
                            param['lr'] = lr
                            param['batch_size'] = batch_size

                            parameter_lstm_list.append(param)

    #-------LSTM直接调节参数------
    for param in parameter_lstm_list:

        index += 1

        layer_num = param['layer_num']
        cell_num = param['cell_num']
        hidden_num = param['hidden_num']
        nb_epoch = param['nb_epoch']
        dropout = param['dropout']
        lr = param['lr']
        batch_size = param['batch_size']

        result = str(index) + ',' + str(layer_num) + ',' + str(cell_num) + ',' + str(hidden_num) + ','+ str(nb_epoch) + \
                 ',' + str(dropout) + ',' + str(lr) + ',' + str(batch_size)
        writer = open(filename_parameter_adjust_result, 'a')
        writer.write(result + '\n')
        writer.close()

        x_train = np.load(filename_x_train)
        y_train = np.load(filename_y_train)
        x_test = np.load(filename_x_test)
        y_test = np.load(filename_y_test)

        lstm_31train_31test(index, x_train, y_train, x_test, y_test, layer_num, cell_num, nb_epoch, dropout, lr,
                            batch_size)



