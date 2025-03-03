import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import modulation_lib
from tqdm import tqdm


lat_vct = np.array(
    [
        [2.51244, 0.000000, 0.000000],
        [1.25622, 2.17583686548, 0.000000],
        [0.000000, 0.000000, 15.0],
    ]
)
orb_sites = np.array(
    [
        [0.00000, 0.00000, 0.25000],
        [0.33333, 0.33333, 0.25],
        # [0.33333, 0.33333, 0.37218],
    ]
)

chi_idx = np.array([22, 0])
trans_idx, sc_index = modulation_lib.cal_chiral_sc_idx(chi_idx)
print(f"trans_idx is {trans_idx[0]}, {trans_idx[1]}")

for i_orb, orb_site in enumerate(orb_sites):
    new_lat_vct, candidate_posi = modulation_lib.make_supercell(
        lat_vct=lat_vct,
        orb_sites=orb_site.reshape(1, 3),
        supercell_index=sc_index,
    )

    reg_lat_vct = np.array(
        [
            [np.linalg.norm(new_lat_vct[0]), 0, 0],
            [0, np.linalg.norm(new_lat_vct[1]), 0],
            [0, 0, new_lat_vct[2, 2]],
        ]
    )

    def costum_shape_func(height: float, period: float) -> sp.Expr:
        x = sp.symbols("x")
        return height * sp.exp(-((x - period / 2) ** 2) / (period * 1.5))

    orb_sites_modi, lat_vct_modi = modulation_lib.chiral_modulation(
        lat_vct=reg_lat_vct,
        orb_sites=candidate_posi[:, :3],
        height=3,
        modi_shape_func=costum_shape_func,
    )

    with open("POSCAR_achiral_hbn_22_0_test", "a") as f:
        for i in range(orb_sites_modi.shape[0]):
            f.write(
                f"{orb_sites_modi[i,0]:>12.8f}{orb_sites_modi[i,1]:>12.8f}{orb_sites_modi[i,2]:>12.8f}\n"
            )

print(lat_vct_modi)
