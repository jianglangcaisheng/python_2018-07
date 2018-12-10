
import scipy.io as sio

data = sio.loadmat('../preproduce/B_D_EII.mat')
B_mu = data['B']
B_mu_local = B_mu[0][0]
c = 1