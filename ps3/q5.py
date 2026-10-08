
import numpy as np
import matplotlib.pyplot as plt


var = np.array([4.0, 3.0, 0.8, 0.1])

#A) ----------------

def computeDR(var, level):
    D_i = np.minimum(var, level)
    R_i = np.zeros_like(var)

    active = var > level
    R_i[active] = 0.5 * np.log2(
        var[active] / level
    )

    D = np.sum(D_i)
    R = np.sum(R_i)

    return D, R, D_i, R_i

#B)----------------------
"""
for x in range(0, 10): #script used to find lambda 1,2
    print(x*.2, computeDR(var, .2*x)[1])

level, R
0.0 inf
0.2 5.114409345247941
0.4 3.6144093452479407
0.6000000000000001 2.736965594166206
0.8 2.11440934524794
1.0 1.792481250360578
1.2000000000000002 1.5294468445267841
1.4000000000000001 1.307054423190336
1.6 1.1144093452479404
1.8 0.9444843438056281
"""
#Choose lambda = 0.8 and 1.2


for level in np.linspace(0.8, 1.0, 2001):
    D, R, Di, Ri = computeDR(var, level)

    if 1.99 < R <= 2:
        best_level = level
        best_D = D
        best_R = R

print(best_level, best_D, best_R)
#lambda = 0.872, D = 2.644, R = 1.9900812102457386
levelChosen = 0.872

#C)---------------------------


levels = np.linspace(0.8, 1.0, 200)
D_values = []
R_values = []

for level in levels:
    D, R, _, _ = computeDR(var, level)
    D_values.append(D)
    R_values.append(R)

D1, R1, _, _ = computeDR(var, 0.8)
D2, R2, _, _ = computeDR(var, 1.0)

plt.plot(D_values, R_values, label="R(D) curve")
plt.plot([D1, D2], [R1, R2], "--", label="line seg")

plt.xlabel("Distortion D")
plt.ylabel("Rate R (bits)")
plt.title("Rate-Distortion Curve")
plt.grid()
plt.legend()
plt.savefig("R(D) curve")

#the line segment is always above the R(D) curve, thus the rate-distortion curve is convex

#D)--------------------------
#Increasing lambda increases D because more distortion is allowed in each component. However, R will be decreases because fewer bits are required to represent the components

#E)----------------------------


D, R, Di, Ri = computeDR(var, levelChosen)

print(f"{'Component':<12}{'Variance':<12}{'Di':<12}{'Ri':<12}{'Encoded'}")

for i in range(len(var)):
    print(f"{i+1:<12}{var[i]:<12.4f}{Di[i]:<12.4f}{Ri[i]:<12.4f}{var[i] > levelChosen}")

print(f"{'Total':<24}{D:<12.4f}{R:<12.4f}")


#components 3 and 4 are not encoded because their variances are less than 0.872