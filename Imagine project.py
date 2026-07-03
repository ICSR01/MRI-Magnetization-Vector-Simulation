# Import required packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Setting parameters - Initial magnetization (A/m), Static magnetic field, Time vector, Gyromagnetic
M0 = 40.0
B0 = 1.0
time = np.linspace(0, 5, 5000)
dt = time[1] - time[0]
time_ms = time * 1000
gyromagnetic_ratio = 2 * np.pi * 42.58e6
omega0 = gyromagnetic_ratio * B0
T1_A, T2_A = 3.0, 0.3
T1_B, T2_B = 1.5, 0.1

# Initial conditions
Mx0, My0, Mz0 = 40, 0, 0

# Definig function bloch_solution using the required Matrix Solution
def bloch_solution(time_array, T1, T2, use_eff_freq=False):
    # Scale frequency for 3D visualization to prevent aliasing, use true frequency for math
    omega = 2 * np.pi * 3 if use_eff_freq else omega0
    
    M = np.zeros((3, len(time_array)))
    M[:, 0] = [Mx0, My0, Mz0]
    
    E1 = np.exp(-dt / T1)
    E2 = np.exp(-dt / T2)
    
    A = np.array([
        [E2 * np.cos(omega * dt), E2 * np.sin(omega * dt), 0],
        [-E2 * np.sin(omega * dt), E2 * np.cos(omega * dt), 0],
        [0, 0, E1]
    ])
    
    B_vec = np.array([0, 0, M0 * (1 - E1)])
    
    for i in range(1, len(time_array)):
        M[:, i] = np.dot(A, M[:, i-1]) + B_vec
        
    Mx = M[0]
    My = M[1]
    Mz = M[2]
    Mxy = np.sqrt(Mx**2 + My**2)
    
    return Mx, My, Mz, Mxy

MxA, MyA, MzA, MxyA = bloch_solution(time, T1_A, T2_A, use_eff_freq=False)
MxB, MyB, MzB, MxyB = bloch_solution(time, T1_B, T2_B, use_eff_freq=False)

MxA_3d, MyA_3d, MzA_3d, _ = bloch_solution(time, T1_A, T2_A, use_eff_freq=True)
MxB_3d, MyB_3d, MzB_3d, _ = bloch_solution(time, T1_B, T2_B, use_eff_freq=True)

# Plotting graphs
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# First graph - Transverse Relaxation
ax1.plot(time_ms, MxyA, color='blue', label='Moiety A (T2=300 ms)')
ax1.plot(time_ms, MxyB, color='brown', label='Moiety B (T2=100 ms)')
ax1.set_title('Transverse (T2) Relaxation')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Transverse magnetization |M_xy| (A/m)')
ax1.legend()
ax1.grid(True)

# Second graph - Longitudinal Recovery
ax2.plot(time_ms, MzA, color='blue', label='Moiety A (T1=3 s)')
ax2.plot(time_ms, MzB, color='brown', label='Moiety B (T1=1.5 s)')
ax2.set_title('Longitudinal (T1) Relaxation')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Longitudinal magnetization M_z (A/m)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# 3D Trajectories
fig = plt.figure(figsize=(10, 10), constrained_layout=True)
ax = fig.add_subplot(111, projection='3d')
ax.plot(MxA_3d, MyA_3d, MzA_3d, color='blue', label='Moiety A')
ax.plot(MxB_3d, MyB_3d, MzB_3d, color='brown', label='Moiety B')
ax.set_xlabel('Mx (A/m)')
ax.set_ylabel('My (A/m)')
ax.set_zlabel('Mz (A/m)')
ax.set_title('Bloch Magnetization Trajectories')
ax.legend()
plt.show()