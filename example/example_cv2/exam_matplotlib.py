
import matplotlib.pyplot as plt
import matplotlib.patches as patches

if 0:
    fig1 = plt.figure(figsize=(1004, 1004))
    print("finish figure")
    ax1 = fig1.add_subplot(111, aspect='equal')
    print("finish add_subplot")
    for i_ractange in range(B8[0].__len__()):
        ax1.add_patch(patches.Rectangle(xy=(B8[0][i_ractange][0], B8[0][i_ractange][1]),
                                        width=B8[0][i_ractange][2] - B8[0][i_ractange][0],
                                        height=B8[0][i_ractange][3] - B8[0][i_ractange][1],
                                        color=C8[0][i_ractange]))
    print("finish add_patch")
    plt.show(fig1)
    input()
if 0:
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    for i_ractange in range(B8[0].__len__()):
        ax1.add_patch \
            (patches.Rectangle(xy=(B8[0][i_ractange][0] / cf.imageWidth, B8[0][i_ractange][1] / cf.imageHeight),
                                        width=B8[0][i_ractange][2] / cf.imageWidth - B8[0][i_ractange][0] / cf.imageWidth,
                                        height=B8[0][i_ractange][3] / cf.imageHeight - B8[0][i_ractange][1] / cf.imageHeight,
                                        color=C8[0][i_ractange]))

    plt.show(fig1)
    input()