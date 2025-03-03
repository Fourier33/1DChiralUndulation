import numpy as np
import sympy as sp
from scipy.integrate import quad
import math

# modulation shape function example
# def costum_shape_func(height: float, periodic: float) -> sp.Expr:
#     x = sp.symbols("x")
#     return height * sp.sin(x * 2 * np.pi / periodic)


def cal_period_l(target_length: float, height: float, modi_shape_func) -> float:
    x = sp.symbols("x")
    eps = 1
    upper_bound = target_length + 1
    lower_bound = 1
    while eps > 1e-5:
        l_f = (upper_bound + lower_bound) / 2
        f = modi_shape_func(height, l_f)
        df = sp.diff(f, x)
        cur_lin_int = quad(sp.lambdify(x, sp.sqrt(1 + df**2)), 0, l_f)
        eps = abs(cur_lin_int[0] - target_length)
        if cur_lin_int[0] > target_length:
            upper_bound = l_f
        else:
            lower_bound = l_f
    return l_f


def cur_lin_integral(
    x_init: float, height: float, period: float, modi_shape_func
) -> float:
    x = sp.symbols("x")
    f = modi_shape_func(height, period)
    df = sp.diff(f, x)
    cur_lin_core = sp.lambdify(x, sp.sqrt(1 + df**2))

    eps = 1
    upper_bound = x_init + 1
    lower_bound = -1
    while eps > 1e-5:
        x_fin = (upper_bound + lower_bound) / 2
        cur_lin_int = quad(cur_lin_core, 0, x_fin)
        eps = abs(cur_lin_int[0] - x_init)
        if cur_lin_int[0] > x_init:
            upper_bound = x_fin
        else:
            lower_bound = x_fin
    return x_fin


def chiral_modulation(
    lat_vct: np.array, orb_sites: np.array, height: float, modi_shape_func
):
    # lat_vct is 3 x 3 array
    # lat_vct[0] is the chiral vector by design

    chi_l = cal_period_l(lat_vct[0, 0], height, modi_shape_func)

    # I need orb vectors to be co-planar with lat_vct[0]
    # currently, this code only applies to true 2d
    # materials, like graphene, not TMDs
    orb_sites_modi = np.zeros_like(orb_sites)

    orb_sites_cart = np.matmul(orb_sites, lat_vct)

    orb_sites_x_shrk = np.zeros(len(orb_sites_cart))
    for i in range(len(orb_sites_cart)):
        orb_sites_x_shrk[i] = cur_lin_integral(
            orb_sites_cart[i, 0], height, chi_l, modi_shape_func
        )

    modi_shape_func_lam = sp.lambdify(
        sp.symbols("x"), modi_shape_func(height, chi_l), "numpy"
    )
    orb_site_z_modi_cart = modi_shape_func_lam(orb_sites_x_shrk)
    orb_site_z_modi = orb_site_z_modi_cart / lat_vct[2, 2] + orb_sites[0, 2]

    orb_sites_modi[:, 2] = orb_site_z_modi
    orb_sites_modi[:, 0] = orb_sites_x_shrk / chi_l
    orb_sites_modi[:, 1] = orb_sites[:, 1]
    lat_vct_modi = lat_vct.copy()
    lat_vct_modi[0, 0] = chi_l
    return orb_sites_modi, lat_vct_modi


def make_supercell(
    lat_vct=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    orb_sites=np.array([[0, 0, 0], [0.5, 0.5, 0.5]]),
    supercell_index=np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]]),
):
    # consider only 2d lattice, i_13=i_23=0, i_3=[0,0,1]
    new_lat_vct = np.matmul(supercell_index, lat_vct)
    inv_supercell_index = np.linalg.inv(supercell_index)

    max_range_x = np.sum(np.abs(supercell_index[:2, 0]))
    max_range_y = np.sum(np.abs(supercell_index[:2, 1]))
    candidate_positions = []
    for i_site, site in enumerate(orb_sites):
        for i in range(-max_range_x, max_range_x + 1):
            for j in range(-max_range_y, max_range_y + 1):
                f_old_shifted = site + np.array([i, j, 0], dtype=float)
                f_new = np.matmul(f_old_shifted, inv_supercell_index)
                if np.all(f_new >= -1e-12) and np.all(f_new < 1 - 1e-12):
                    f_new_indexed = np.concatenate(
                        (f_new, np.array([i, j, 0, i_site + 1]))
                    )
                    candidate_positions.append(f_new_indexed)
    return new_lat_vct, np.vstack(candidate_positions)


def cal_chiral_sc_idx(chi_idx):
    d_r = math.gcd(2 * chi_idx[1] + chi_idx[0], 2 * chi_idx[0] + chi_idx[1])
    trans_idx = np.array(
        [(2 * chi_idx[1] + chi_idx[0]) / d_r, -(2 * chi_idx[0] + chi_idx[1]) / d_r],
        dtype=int,
    )
    sc_idx = np.array(
        [
            [chi_idx[0] * 1, chi_idx[1] * 1, 0],
            [trans_idx[0] * 1, trans_idx[1] * 1, 0],
            [0, 0, 1],
        ]
    )
    return trans_idx, sc_idx
