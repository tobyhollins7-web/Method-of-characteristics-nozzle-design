# mach_angle_and_prandlt_meyer.py
import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from config import GAMMA, NUM_INTEGRATION_POINTS, MACH_NUMBER

def calculate_mach_angle(mach_number):
    # function that calculates the mach angle number, given a mach number, returns the angle in degrees
    if mach_number < 1:
        raise ValueError('mach_number must be greater than 1')
    mach_angle_rad = np.asin(1 / mach_number)
    return np.rad2deg(mach_angle_rad)

def prandtl_meyer_function(mach_number, gamma=GAMMA,num_integration_steps=NUM_INTEGRATION_POINTS, baseline_mach_number=1):
    # prandtl meyer numerical integration, using Simpson's, function
    # NOTE: always has default values of GAMMA, NUM_INTEGRATION_POINTS from config.py
    if mach_number - baseline_mach_number < 0:
        raise ValueError(f'mach_number must be greater than baseline_mach_number. {mach_number:.1f} <'
                         f'{baseline_mach_number:.1f}')
    ms = np.linspace(baseline_mach_number, mach_number, num_integration_steps)  # inclusive for end points

    integrand = np.sqrt(ms ** 2 - 1) / (ms * (1 + (gamma - 1) * ms ** 2 / 2))
    theta = np.rad2deg(simpson(integrand, ms))
    return theta

def plot(M, gamma=GAMMA, M_baseline=1, num_points=int(1e6)):
    # plot used to show the integrand within the prandtl meyer function
    # this function was used to visually show a curved relationship betweenn nu and Mach number
    # hence Simpson's rule was used (quadratic areas, better fit)
    ms = np.linspace(M_baseline, M, num_points)
    ys = np.sqrt(ms ** 2 - 1) / (ms * (1 + (gamma - 1) * ms ** 2 / 2))

    plt.figure(figsize=(8, 8), dpi=100)
    plt.plot(ms, ys, label='$\\mathcal{M}$')
    plt.legend(loc='best')
    plt.xlim(M_baseline * 0.95, M * 1.05)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    mach_angle = calculate_mach_angle(MACH_NUMBER)
    nu = prandtl_meyer_function(MACH_NUMBER)
    print(f"="*50)
    print(f"@ conditions: M={MACH_NUMBER}, gamma={GAMMA}")
    print(f"mach_angle={mach_angle:.6f}, nu={nu:.6f}")  # What are my Mach and nu values @flow conditions
    M = MACH_NUMBER
    g = GAMMA
    ratio = 1 / M * (2 / (g + 1) * (1 + (g - 1) / 2 * M ** 2))**((g + 1)/ (2 * (g - 1)))
    print(f"ratio={ratio:.6f}")  # A/A* ratio from Quasi 1D