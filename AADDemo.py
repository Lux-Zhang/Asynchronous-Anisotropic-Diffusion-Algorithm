import matplotlib.pyplot as plt

from scipy import io
from AsynchronousAD import *
from scipy.optimize import leastsq

def lorentz(p, x):
    return p[0] / ((x - p[1]) ** 2 + p[2])


def errorfunc(p, x, z):
    return z - lorentz(p, x)


def lorentz_fit(x, y):
    p3 = ((max(x) - min(x)) / 10) ** 2
    p2 = (max(x) + min(x)) / 2
    p1 = max(y) * p3
    p0 = np.array([p1, p2, p3], dtype=np.float)  # Initial guess
    solp, ier = leastsq(errorfunc, p0, args=(x, y), maxfev=50000)
    return solp[1]

if __name__ == "__main__":

    SPN = 10
    KernelSize = 5
    option = 2
    GaussSigmaAAD = 5
    Thres = 0.16514297461253982 # 1000 * standard deviation of noise
    EdgeCutoffLow = 0.40
    EdgeCutoffHigh = 0.90
    NoiseCutoff = 0.07
    iterNumber = 64 # SNR = 20

    data = io.loadmat('./data.mat')
    BGS = data['data_G'].T
    BGS = np.array(BGS)
    BGS = BGS.astype(np.float32)

    F = asyanisodiff2D
    BGS_new = F(BGS, IterNumber=iterNumber, GaussSigma=GaussSigmaAAD, Kappa=Thres, KernelSize=KernelSize, Option=option,
                NoiseCutoff=NoiseCutoff, EdgeCutoffLow=EdgeCutoffLow, EdgeCutoffHigh=EdgeCutoffHigh, EdgeKeeping='yes')

    #lorentz-fit
    fre = np.linspace(10.750, 10.950, 200)  #frequency
    lortz_data = []
    for each in BGS_new:
        lortz_fit = lorentz_fit(fre, each)
        lortz_data.append(lortz_fit)
    BFS_lortz = np.array(lortz_data)

    #plot
    x = np.linspace(1, 2549,2550)
    plt.plot(x, BFS_lortz)
    plt.show()