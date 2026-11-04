"""SI parameters used in the current AARB_Article_v1 comparison.
Vehicle: Vu et al. (2017), Table 3. Hydraulic actuator: Table 1.
Geometry assumptions are separated at the bottom; no parameters are fitted at runtime.
"""
# Mass (kg), lengths (m), speed (m/s), gravity (m/s^2).
ms, muf, mur, m = 12487., 706., 1000., 14193.
lf, lr, lw, h, rh, hu = 1.95, 1.54, .93, 1.15, .83, .53
v, g = 70 / 3.6, 9.81
# Inertias (kg m^2); roll stiffnesses (N m/rad); damping (N m s/rad).
Ixx, Ixz, Izz = 24201., 4200., 34917.
kf, kr, ktf, ktr = 380000., 684000., 2060000., 3337000.
bf = br = 100000.
Cf, Cr = 582000., 783000.
# Eq. (11) uses mu=1 for this reference case.
mu = 1.
# Hydraulic parameters: area m^2, Kx m^2/s, KP m^5/(N s), Vt m^3,
# bulk Pa, tau s, Kv m/A; zero leakage coefficient.
area, Kx, KP, Vt, bulk, tau, Kv = .0123, 2.5, 4.2e-11, .0014, 6.89e6, .01, .0239
Ctp = 0.
# Table 3: passive torsional stiffness (N m/rad).
kAOf, kAOr = 10730., 15480.
# Supplementary modelling assumptions used for the current comparison.
# L_ACT=.50 was selected after an exploratory comparison with Figure 8;
# it is not an explicit numerical entry in the reference paper's tables.
L_ACT = .50
T_A, T_B, ARM_LENGTH = .93, .20, .18
DURATION = 10.
DT = .001