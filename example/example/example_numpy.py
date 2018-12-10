
import numpy as np

# 例程
if 0:
    a = np.arange(9).reshape(3, 3)
    b = np.eye(3, 3)

    print(a)
    print(b)

    b[0:2, 0] = a[0:2, 1]
    print(b)

# 方块赋值
if 0:
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([[11, 12], [14, 15]])
    print(a)
    print(b)

    print(np.size(b, 0))
    print(np.size(b, 1))

    a[0:np.size(b, 0), 0:np.size(b, 1)] = b     # 重点！！！
    print(a)

# 向量赋值矩阵
if 0:
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([11, 12, 13])
    print(a)
    print(b)

    print(np.size(b, 0))

    a[0:np.size(b, 0), 0] = b     # 可以赋值到矩阵的行列
    print(a)

# shape不等，不能：赋值。（要广播）
if 0:
    a = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10], [7, 8, 9, 10]])
    b = np.array([[11, 12], [14, 15]])
    print(a)
    print(b)

    print(np.size(b, 0))
    print(np.size(b, 1))

    a[:, :] = b     # 重点！！！
    print(a)

# 交换维度
if 0:
    a = np.zeros([2, 3, 4])
    print(a.shape)

    b = np.transpose(a, [1, 2, 0])
    print(b.shape)

# 存储
if 0:
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    np.savez("important_temp.npz", a=a, b=b)

# 读取
if 0:
    npzfile = np.load("important_temp.npz")
    a = npzfile['a']
    print(a)

# 拼接
if 1:
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    c0 = np.concatenate([a, b], axis=0)
    # [[1 2]
    #  [3 4]
    #  [5 6]
    #  [7 8]]
    c1 = np.concatenate([a, b], axis=1)
    # [[1 2 5 6]
    #  [3 4 7 8]]
    c = np.concatenate((np.reshape(a[:, 1], [-1, 1]), np.reshape(a[:, 0], [-1, 1])), axis=1)
    print(a)
    print(b)
    print(c0)
    print(c1)
    print(c)

# 矩阵乘法
if 0:
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[1, 1], [1, 1]])
    print(a)
    print(b)
    c = np.matmul(a, b)
    d = a * b
    print(c)
    print(d)

# 数组求和
if 0:
    a = np.ones([2, 2, 2])
    sum = np.sum(a)
    print(sum)

# 乘方
if 0:
    a = np.array([[1, 2], [3, 4]])
    print(a)

    b = a * a
    print(b)
    c = np.power(a, 3)      # 点乘
    print(c)

# 开方
if 0:
    a = 100
    b = np.sqrt(a)
    print(b)

# logic
if 0:
    a = np.array([[2.5, 1.5], [2.5, 3.5]])
    aT = a > 2


    x = np.array([[10., 10.], [10., 10.]])
    y = np.array([[100., 100.], [100., 100.]])

    #赋值方式，右边是向量
    a[aT] = x[aT]
    a[np.logical_not(aT)] = y[np.logical_not(aT)]
    print(a)

# logic .*
if 0:
    a = np.array([[2.5, 1.5], [2.5, 3.5]])
    aT = a > 2

    print(a)
    print(aT)

    x = np.array([[10., 10.], [10., 10.]])
    y = np.array([[100., 100.], [100., 100.]])

    a = y * aT + x * np.logical_not(aT)
    print(a)

# min
if 0:
    a = np.array([[2.5, 1.5], [2.5, 3.5]])
    print(a)

    b = np.minimum(a, 2. * np.ones(shape=np.shape(a)))
    print(b)

    c = np.minimum(a, 2.)
    print(c)

