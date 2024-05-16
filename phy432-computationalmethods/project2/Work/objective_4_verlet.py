import numpy as np
import matplotlib.pyplot as plt


# Initialize variables
# constants
m_e = 9.11e-31
c = 2.998e8
mu0 = 1.257e-6
eps0 = 8.854e-12
e = 1.6022e-19

# Table 2
R = 0.1
L = 0.2
z1 = 0

# lorentz force function
def F_lorentz(v, B=0):
    return -1.6022e-19 * (np.cross(v, B))

def Bz_B0(z, z1, z2, R):
    return 0.5 * (((z - z1) / np.sqrt((z-z1)**2 + R**2)) - ((z-z2) / np.sqrt((z-z2)**2 + R**2)))

def Br_B0(z, z1, z2, r, R):
    return (0.25 * r * R**2) * (((z-z2)**2 + R**2)**(-3/2) - ((z-z1)**2 + R**2)**(-3/2))

# z-component magnetic field
def B_field(p, B0=1e-3, R=0.1, z1=0, z2=0.2):
    """Returns B field vector at position p(x,y,z) given B0 and lens radius R
    """
    x = p[0]
    y = p[1]
    z = p[2]
    r = np.sqrt(x**2 + y**2)
    
    # B field components
    Br = B0 * Br_B0(z, z1, z2, r, R)
    Bz = B0 * Bz_B0(z, z1, z2, R)
    
    # Unit vectors
    ez = np.array([0, 0, 1])
    
    # In case r = 0
    if r == 0:
        er = np.array([0, 0, 0])
    else:
        er = np.array([x, y, 0]) / r
        
    return Br*er + Bz*ez

B0 = B_field(p = np.array([3/4*R, 0, -0.3]))

# Solenoid
L_sol = 0.75
B_0_sol = B_field(p = np.array([3/4*R, 0, -0.3]), B0 = 0.0035, 
                  R = 0.1, z1 = 0, z2 = 0.75)

def integrate_EOM(r0, v0, z2, B0=1e-3, R=0.1, tmax=80e-9, h=1e-10):
    Nsteps = int(tmax/h)
    t_range = h * np.arange(Nsteps)
    
    y = np.array([r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]], dtype=np.float64)
    
    positions = [[y[0], y[1], y[2]]]
    velocities = [[y[3], y[4], y[5]]]
    
    # Standard form force function
    def f(t, y, m=m_e):
        # y = [x, y, z, vx, vy, vz]
        r = np.array([y[0], y[1], y[2]])
        v = np.array([y[3], y[4], y[5]])
        B = B_field(r, z2 = z2)
        F = F_lorentz(v, B)
        return np.array([y[3], y[4], y[5], F[0]/m, F[1]/m, F[2]/m])
    
    # rk4
    for i, t in enumerate(t_range[:-1]):
        k1 = f(t, y)
        k2 = f(t + 0.5*h, y + 0.5*h*k1)
        k3 = f(t + 0.5*h, y + 0.5*h*k2)
        k4 = f(t + h, y + h*k3)
        y += h/6 * (k1 + 2*k2 + 2*k3 + k4)
        positions.append([y[0], y[1], y[2]])
        velocities.append([y[3], y[4], y[5]])
        
    return np.array(positions), np.array(velocities), t_range

# Function Execute
if __name__ == "__main__":
    # Lens vectors
    r_0 = [3/4*R, 0, -0.3]
    v_0 = [0, 0, 0.06*c]

    r, v, t = beam_simulation(r_0, v_0, z2 = 0.2)
    
    r_sol, v_sol, t_sol = beam_simulation(r_0, v_0, B0 = 0.0035, R = 0.1, z2 = 0.75)
    
    
    # Solenoid vectors
    plt.plot(t, r[:, 0])
    plt.plot(t, r[:, 1])
    plt.show()
    
    plt.plot(t, r[:, 2])
    plt.show()
    
    plt.plot(r[:, 2], r_sol[:, 0])
    plt.plot(r[:, 2], r_sol[:, 1])
    plt.show()
    # Focal length isnt really possible to estimate because the values retrieved are exceedingly large.
