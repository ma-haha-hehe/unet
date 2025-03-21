import argparse


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model): #计算神经网络模型中可训练参数的总数量
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset() #初始化该对象的所有统计量

    def reset(self):
        self.val = 0 #存储当前数值
        self.avg = 0 #存储平均值
        self.sum = 0 #存储累加总和
        self.count = 0 #存储的数据的个数

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
