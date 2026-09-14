import numpy as np
import matplotlib.pyplot as plt

#plotter function
def plotComplex(array, title):
    x = [ele.real for ele in array]
    y = [ele.imag for ele in array]
    ax = plt.subplot()
    ax.scatter(x,y)
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel("Real")
    plt.ylabel("Imag")
    plt.title(title)
    plt.grid()
    plt.savefig((title + ".jpg"))
    plt.close()

#A)
# 1. QPSK
dmin = 1.0

qpsk = (dmin / np.sqrt(2)) * np.exp(1j * (np.pi/4 + np.arange(4) * np.pi/2))

# 2. 8-PSK
r_8psk = dmin / (2 * np.sin(np.pi / 8))
psk8 = r_8psk * np.exp(1j * np.arange(8) * np.pi / 4)

# 3. 16-QAM
coords_16 = np.array([-1.5, -0.5, 0.5, 1.5]) * dmin
I16, Q16 = np.meshgrid(coords_16, coords_16)
qam16 = (I16 + 1j * Q16).flatten()

# 4. Cross 32-QAM
coords_36 = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]) * dmin
I36, Q36 = np.meshgrid(coords_36, coords_36)
#print(np.meshgrid(coords_36, coords_36))
grid36 = (I36 + 1j * Q36).flatten()


# Exclude the 4 corners: abs(z)^2 == 2.5^2 + 2.5^2 = 12.5
qam32 = grid36[np.abs(grid36)**2 < 12.4]

plotComplex(qpsk, "qpsk")
plotComplex(psk8, "psk8")
plotComplex(qam16, "qam16")
plotComplex(qam32, "qam32")

testarr = np.array([[2+1j,2+1j,2,2],[2,2,3,2],[2,2,2,2]])
print(testarr)
print(np.absolute( testarr))
#Compute Energy per bit: [ 1/log2(M) ]* Es

#B)

def computeEb(array):
    M = array.size
    Es = (1/M) * np.sum(np.power(np.absolute(array), 2))
    k = np.log2(M)
    return ((1/k) * Es)


Eb4 = computeEb(qpsk)
Eb8 = computeEb(psk8) 
Eb16 = computeEb(qam16)
Eb32 = computeEb(qam32)

#C)
def computeN(array):
    return (np.log2(array.size)/ 2) #dimension for all is 2

nqpsk = computeN(qpsk)
n8psk = computeN(psk8)
