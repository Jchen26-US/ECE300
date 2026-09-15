import numpy as np
import matplotlib.pyplot as plt

#Question 1

#plotter function
def plotComplex(array, title):
    x = [ele.real for ele in array]
    y = [ele.imag for ele in array]
    ax = plt.subplot()
    ax.scatter(x,y)
    ax.set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.grid()
    plt.savefig((title + ".jpg"))
    plt.close()

#A) --------------------------------
# 1. QPSK
dmin = 1

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

print("qpsk-----------------------\n", qpsk)
print("psk8-----------------------\n", psk8)
print("qam16----------------------\n", qam16)
print("qam32----------------------\n", qam32)

#confirmation plots
plotComplex(qpsk, "qpsk")
plotComplex(psk8, "psk8")
plotComplex(qam16, "qam16")
plotComplex(qam32, "qam32")

#testarr = np.array([[2+1j,2+1j,2,2],[2,2,3,2],[2,2,2,2]])
#print(testarr)
#print(np.absolute( testarr))
#Compute Energy per bit: [ 1/log2(M) ]* Es

#B)-------------------------------

def computeEb(array):
    M = array.size
    Es = (1/M) * np.sum(np.power(np.absolute(array), 2))
    k = np.log2(M)
    return ((1/k) * Es)


Eb4 = computeEb(qpsk)
Eb8 = computeEb(psk8) 
Eb16 = computeEb(qam16)
Eb32 = computeEb(qam32)
print("Energy per bit [QPSK, 8PSK, 16QAM, 32QAM] : ",Eb4, Eb8, Eb16,Eb32)
#Energy per bit [4,8,16,32] :  0.25 0.5690355937288493 0.6250000000000001 1.0
#C)-------------------------------------
def computeN(array): 
    return (np.log2(array.size)/ 2) #dimension for all is 2

nqpsk = computeN(qpsk)
n8psk = computeN(psk8)
n16qam = computeN(qam16)
n32qam = computeN(qam32)
print("bits/ dimension [QPSK, 8PSK, 16QAM, 32QAM] :", nqpsk, n8psk, n16qam, n32qam)
#bits/ dimension [4, 8, 16, 32] : 1.0 1.5 2.0 2.5

#D) ----------------------------
#The most power efficient is QPSK with an energy per bit of .25

#E) ------------------------------
#The most spectrally efficient is 32QAM with 2.5 bits per dimension

#F)????????????????????
#


#Question2 part (f)
#X(f) = A/2W [ 1 + cos(pif/W)] II(f/2W)
#sinc(E) = sin(piE)/piE
#x(t) = A [ sinc(2Wt)  + 1/2 sinc(2W(t+1/2W)) + 1/2 sinc(2W(t - 1/2W))]
A = 1
W = 1
#change 3rd parameter for higher definition, I used 
f = np.linspace(-2*W, 2*W, 1_000_000)
t = np.linspace(-4*W, 4*W, 1_000_000)
def rect(x, W):
    return np.where(np.abs(x) <= W, 1, 0)

XF  = (A/(2*W)) * (1 + np.cos(np.pi * f / (W))) * rect(f, W)

def sinc(x):
    return (np.sin(np.pi * x)/(np.pi * x))

xt = A * (sinc(2*W*t) + 1/2 * sinc(2*W*(t + 1/(2*W))) + 1/2 * sinc(2*W*( t - 1/(2*W))))

fig1,ax1= plt.subplots()

ax1.plot(f,XF)
plt.grid()
plt.title("X(f)")
plt.savefig("X(f).jpg")

fig2,ax2  = plt.subplots()
ax2.plot(t,xt)
plt.grid()
plt.title("x(t)")
plt.savefig("x(t).jpg")