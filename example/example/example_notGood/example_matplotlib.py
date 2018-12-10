
import matplotlib.pyplot as plt
import matplotlib.patches as patches


fig1 = plt.figure()
ax1 = fig1.add_subplot(111, aspect='equal')

ax1.add_patch(patches.Rectangle(xy=(0.1, 0.1), width=0.5, height=0.5, color=(1, 0, 0)))
ax1.add_patch(patches.Rectangle(xy=(0.23, 0.45), width=0.5, height=0.5, color=[0, 0, 1]))

plt.show(fig1)
