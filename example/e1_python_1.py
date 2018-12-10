
import numpy as np

# 列表
if 0:
    a = []
    a.append(8)
    a.append(6)
    for i in range(a.__len__()):
        print(a[i])

# 元祖
if 0:
    a = (4, 5, 9)
    b = (4,)
    for i in range(b.__len__()):
        print(b[i])

# 乘方
if 0:
    print(pow(2.1, 3.1))

# 列表引用
if 0:
    def func(x):
        x.append(1)
    def func2(x):
        x[0] = 'a'
    B = []
    print (B)

    func(B)
    print(B)

    func2(B)
    print(B)



# model
if 0:
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(a)

# exp
if 0:
    a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    print(a)

    # .exp - 1
    b = np.expm1(a)
    print(b)

    # .exp
    c = np.exp(a)
    print(c)

    # .exp floor?
    d = np.exp2(a)
    print(d)

# trace
if 0:
    a = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    print(a)

    print(np.trace(a))

# *
if 0:
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(a)

    print('\n')
    print(np.multiply(a, 2))

    print('\n')
    print(a * 2)

    print('\n')
    print(np.divide(a, 2))

    print('\n')
    print(a / 2)

# *
if 1:
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(a)

    flag = np.sum(a, axis=1) > 14.
    print(flag)

    c = a[flag]
    print(c)

    # [[1 2 3]
    #  [4 5 6]
    #  [7 8 9]]
    # [False  True  True]
    # [[4 5 6]
    #  [7 8 9]]