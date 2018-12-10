
import numpy as np

def mkdir(path):
    import os
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([1, 2])


mkdir('tmp')
# 当前目录
np.savez('tmp/123.npz', a=a, b=b)


data = np.load('tmp/123.npz')
print(data['a'])
print(data['b'])
data.close()
