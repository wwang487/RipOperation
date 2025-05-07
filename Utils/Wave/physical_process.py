import numpy as np

# some global parameter
g = 9.80665
pi = 3.141592653


def stockton_runup(H0: float, L0: float, beta: float):
    """stockton wave runup equation

    Arguments:
        H0 -- offshore wave height(m)
        L0 -- offshore wave length(m)
        beta -- beach slope (rads)

    Returns:
        wave runup for 2% of maximum incoming wave
    """

    R = 1.1 * (0.35 * beta * np.sqrt(H0 / L0) +
               np.sqrt(H0 * L0 * (0.563 * beta**2 + 0.004)))

    return R


def komar_lst(hb: float, alpha: float, rho: float = 1) -> float:
    """komar longshore sediment transprot rate equation

    Arguments:
        hb -- breaking wave height (m)
        alpha -- breaking wave angle relative to shore/bathymetry
        (degree)

    Keyword Arguments:
        rho -- water density (gr/cm3) (default: {1})

    Returns:
        longshore sediment transport rate (m3/day)
    """

    Q_cerc = 1.1 * rho * g**(1.5) * np.sin(np.deg2rad(alpha)) * \
        np.cos(np.deg2rad(alpha)) * hb**2.5
    return Q_cerc


def closure_depth(He: float = None, Te: float = None) -> float:
    """
    closure depth

    Args:
        He (float, optional): offshore wave height(m). Defaults to None.
        Te (float, optional): offshore wave period(m). Defaults to None.

    Returns:
        float: closure depth(m)
    """
    hc = 2.28 * He - 68.5 * (He**2 / g / Te**2)
    return hc


# wave dispersion relationship
def wave(d, T) -> list:
    """

    :param d: depth (m)
    :param T: wave period (s)
    :return: wave length (m), wave speed(m/s)
    """
    if d > 0.5 * 1.56 * T**2:
        return 1.56 * T**2, 1.56 * T
    L0 = 0
    L1 = g * T**2 / 2 / pi

    while np.abs(L0 - L1) > 1e-1:
        L0 = L1
        L1 = g * T**2 / 2 / pi * np.tanh(2 * pi * d / L0)

    L = L1
    C = L1 / T

    return L, C


def wave_routing(h: float = None,
                 alpha0: float = None,
                 T: float = None,
                 d0: float = 40.0,
                 kb: float = 0.8,
                 opt: bool = False) -> list:
    """
    wave routing method: calculating the breaking wave heights and depth
     using binary search

    Args:
        h (float, optional): offshore wave height. Defaults to None.
        alpha0 (float, optional): offshore wave angle (relative to shoreline
        or bathymetry normal line). Defaults to None.
        T (float, optional): offshore wave period. Defaults to None.
        d0 (float, optional): offshore wave depth. Defaults to 40.0.
        kb (float, optional): breaking criteria. Defaults to 0.8.
        opt (bool, optional): cython optimization for acceleration. Defaults
         to False.

    Returns:
        list: [breaking depth, breaking height, breaking angle,
         breaking vecolity, offshore wave length]
    """
    # import cython module if opt is true
    global wave
    if opt is True:
        from cy_extension.wave_cy import wave

    # define the lambda for n
    def n(L, d):
        return 0.5 * \
            (1 + (2 * 2 * pi / L * d / np.sinh(2 * 2 * pi / L * d)))

    # define two ptrs at offshore and nearshore depth
    nearshore_ptr = 1e-2
    offshore_ptr = d0

    # wave dispersion relationship (offshore)
    L0, c0 = wave(d0, T)
    n0 = n(L0, d0)
    cg0 = c0 * n0

    # iterate to find the breaking site
    while offshore_ptr - nearshore_ptr > 1e-2:

        # find the current depth at mid of ptrs
        cur_depth = (nearshore_ptr + offshore_ptr) / 2

        # calculate the wave climate at current depth
        L, c = wave(cur_depth, T)
        cur_n = n(L, cur_depth)
        cg = c * cur_n

        # refraction + shoaling
        kr = (np.cos(np.deg2rad(alpha0)) /
              np.sqrt(1 - (c / c0 * np.sin(np.deg2rad(alpha0)))))**0.5
        ks = np.sqrt(cg0 / cg)
        cur_height = h * kr * ks
        alpha = np.degrees(np.arcsin(c / c0 * np.sin(np.deg2rad(alpha0))))

        # calculate h/d
        cur_k = cur_height / cur_depth

        # update ptrs
        if cur_k < kb:  # forward
            offshore_ptr = (offshore_ptr + nearshore_ptr) / 2
        else:  # backward
            nearshore_ptr = (offshore_ptr + nearshore_ptr) / 2

    return [cur_depth, cur_depth, alpha, c, L0]


def wave_routing_by_depth(h0: float,
                          alpha0: float,
                          T: float,
                          d0: float,
                          target_depth: float = 2,
                          breaking_criteria: float = 0.8,
                          steps=100) -> list:
    """wave routing

    Args:
        h0 (float): offshore wave height (m)
        alpha0 (float): offshore wave incident angle to shoreline (deg)
        T (float): offshore wave period (s)
        d0 (float): station depth (m)
        target_depth (float, optional): estimate water depth (m). Defaults to 2.
        breaking_criteria (float, optional): breaking criteria. Defaults to 0.8.
        steps (int, optional): calculating steps. Defaults to 100.

    Returns:
        list: [wave height (m), propagation time (s), wave angle to shoreline(deg)]
    """
    def n(L, d):
        return 0.5 * \
            (1 + (2 * 2 * pi / L * d / np.sinh(2 * 2 * pi / L * d)))

    def goda(depth, l0, slope):
        return l0/7 * (1-np.exp(pi*depth/l0*(16.21*slope*slope-7.07*slope-1.55)))

    def om(depth, l, slope):
        return l/7*np.tanh(2*pi*depth/l*(-11.21*slope*slope+5.01*slope+0.91))

    L0, c0 = wave(d0, T)
    n0 = n(L0, d0)
    cg0 = c0 * n0

    depth_shallow = np.linspace(10, target_depth, int(steps/2))
    step_shallow_dist = (6000-357)/(steps/2)
    depth_deep = np.linspace(d0, 10, steps-int(steps/2))
    step_deep_dist = 357/(steps/2)
    depths = np.concatenate([np.array(depth_deep),
                            np.array(depth_shallow)], axis=0)

    total_time = 0
    for i, cur_depth in enumerate(depths):
        L, c = wave(cur_depth, T)
        cur_n = n(L, cur_depth)
        cg = c * cur_n

        # refraction + shoaling
        cosalpha = np.sqrt(1 - (c/c0 * np.sin(np.deg2rad(alpha0)))**2)
        # alpha = np.degrees(np.arcsin(c / c0 * np.sin(np.deg2rad(alpha0))))
        alpha = np.degrees(np.arccos(cosalpha))
        #print(alpha)
        kr = (np.cos(np.deg2rad(alpha0)) /
              cosalpha)**0.5

        ks = np.sqrt(cg0 / cg)
        # print(kr, ks)
        # kr = 1
        cur_height = h0 * kr * ks

        if i < len(depths)/3:
            dt = step_deep_dist/c
            hb = goda(cur_depth, L0, 0.0053)
        else:
            dt = step_shallow_dist/c
            hb = goda(cur_depth, L0, 0.028)

        total_time += dt
        if cur_height > breaking_criteria:
        #if hb/cur_depth > breaking_criteria:
            print(hb, cur_depth)
            cur_height = hb
            break

    return cur_height, total_time, alpha


# critical boundary velocity
def u_wle(vis: float = None,
          D_50: float = None,
          rho_s: float = None,
          T_0: float = None) -> float:
    """critical boundary velocity

    Args:
        vis (float, optional): velocity(m/s). Defaults to None.
        D_50 (float, optional): particle size(mm). Defaults to None.
        rho_s (float, optional): relative density(kg/m3). Defaults to None.
        T_0 (float, optional): offshore wave period(s). Defaults to None.

    Returns:
        float: critical boundary velocity
    """

    # scaled dimensionless immersed sediment weight
    s_star = D_50 / np.sqrt(rho_s * g * D_50) / 4 / vis

    # some coefficients
    T_r = 159 * s_star**(-1.3) * D_50 / vis
    C = 2.53 * vis * s_star**0.92 / 4 / D_50

    # critical boundary velocity
    u = 2 * pi * C * (1 + 5 * (T_r / T_0)**2)**(-0.25)

    return u


# Bailard-Inman cross-shore sediment transport
def bailard_cst(H0: float = None,
                beach_slope: float = None,
                repose_slope: float = None,
                T: float = None,
                L0: float = None,
                ws: float = 0.15,
                rho: float = 1.0):
    """Bailard-Inman crosshore sediment transport

    Args:
        H0 (float, optional): offshore wave height(m). Defaults to None.
        beach_slope (float, optional): beach slope. Defaults to None.
        repose_slope (float, optional): sand repose slope. Defaults to None.
        T (float, optional): wave period(s). Defaults to None.
        L0 (float, optional): wave length(m). Defaults to None.
        ws (float, optional): settling velocity(m/s). Defaults to 0.15.
        rho (float, optional): relative density(kg/m3). Defaults to 1.0.

    Returns:
        float: crosshore sediment transport
    """

    # direction criterion
    criterion = g * H0 * beach_slope * T / L0 / ws

    # cross shore parameter
    psi1 = 0.303 - 0.00144 * H0
    psi2 = 0.603 - 0.00510 * H0
    sigma_u = 0.458 - 0.00157 * H0
    u_m = 31.9 + 0.403 * H0
    u3_star = 0.548 + 0.000733 * H0
    u5_star = 1.5 + 0.00346 * H0

    # grain parameter
    epsilon_b = 0.2
    epsilon_s = 0.025
    cd = 1

    # cross shore transport
    Q = rho * cd * u_m**3 * (
        epsilon_b / repose_slope *
        (psi1 + 0.67 * sigma_u - beach_slope / repose_slope * u5_star) +
        u_m / ws * (psi2 + sigma_u * u3_star -
                    u_m / ws * epsilon_s * u5_star * beach_slope))

    if criterion > 0.5:  # offshore
        return Q
    else:
        return -Q


# Sea level change
def brunn_slc(W: float = None,
              db: float = None,
              B: float = None,
              Hb: float = None,
              S: float = None) -> float:
    """Sea level change calculation

    Args:
        W (float, optional): beach width(m). Defaults to None.
        db (float, optional): breaking depth(m). Defaults to None.
        B (float, optional): berm height(m). Defaults to None.
        Hb (float, optional): breaking height. Defaults to None.
        S (float, optional): sea level change(m). Defaults to None.

    Returns:
        float: sea level change indicator
    """
    return (S + 0.068 * Hb) * W / (db + B)


# land erosion
def land_erosion(S: float = None,
                 Lp: float = None,
                 p: float = 0.5,
                 B: float = None,
                 hc: float = None) -> float:
    """land erosion calculation

    Args:
        S (float, optional): sea level change. Defaults to None.
        Lp (float, optional): beach profile length. Defaults to None.
        p (float, optional): _description_. Defaults to 0.5.
        B (float, optional): berm height(m). Defaults to None.
        hc (float, optional): closure dpeth(m). Defaults to None.

    Returns:
        float: land erosion indicator
    """
    return S * Lp / (p + 0.1) / (B + hc + 0.1)


def wind_setup(Hb: float, T: float, mb_ns: float, hb: float,
               station_depth: float) -> float:
    """wind setup calculation

    Arguments:
        Hb -- breaking wave height (m)
        T -- wave period (s)
        mb_ns -- beach/nearshore slope (radius). I am not sure beach
         or nearshore
        hb -- breaking depth
        station_depth -- offshore station depth/ deep water depth

    Returns:
        wind setup (m)
    """
    # for depth-limited wave breaking, obtained from Sorenson 1997
    gamma = 0.9

    # offshroe wave length
    L0, _ = wave(station_depth, T)

    # some parameters I don't kwon :(
    k = (2 * pi) / L0
    setdown = -(1 / 8) * ((Hb**2 * k) / (np.sinh(2 * k * hb)))
    mwl_slope = (1 + (8 / (3 * (gamma)**2))**-1) * mb_ns
    dist2shore = (hb - setdown) / mb_ns

    # setup results
    setup = -setdown + dist2shore * mwl_slope
    return setup
