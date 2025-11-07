# nozzle_design.py
import numpy as np
import matplotlib.pyplot as plt
from mach_angle_and_prandtl_meyer import calculate_mach_angle, prandtl_meyer_function
from config import (MACH_NUMBER, NUM_IFT_ENTRIES, NUM_EXPANSION_FANS, EXPANSION_FANS_FROM, X_START,
                    Y_START, IFT_JSON, OVERWRITE, CENTRE_LINE_Y)
from scipy.interpolate import interp1d
import json
from tabulate import tabulate

data = {}  # data dictionary, eventually containing intermediary steps for MoC.

def create_nu_mach_dict(mach_start, mach_end, num_points=1000):
    # creates IFT for global GAMMA
    mach_numbers = np.linspace(mach_start, mach_end, num=num_points)
    nu  = np.array([prandtl_meyer_function(M) for M in mach_numbers])
    order = np.argsort(nu)
    return {
        "M": mach_numbers[order].tolist(),
        "nu": nu[order].tolist(),
    }

# data handling functions (JSON data base)
def save_as_json(data, fname, overwrite=False):
    if not overwrite:
        return
    with open(fname, 'w') as outfile:
        json.dump(data, outfile, indent=4)
    print(f"Wrote {fname}. N={len(data['M'])} samples.")

def load_IFT_JSON(fname):
    with open(fname) as f:
        data = json.load(f)
    M = np.array(data['M'])
    nu = np.array(data['nu'])
    nu_to_M = interp1d(nu, M, kind="linear", fill_value="extrapolate", assume_sorted=True)
    return {"M": M, "nu": nu, "nu_to_M": nu_to_M}

# computational helper functions
def compute_mach_from_nu(nu_target, nu_mach_dict):
    nu_min, nu_max = nu_mach_dict["nu"][0], nu_mach_dict["nu"][-1]
    nu_chosen = float(np.clip(nu_target, nu_min, nu_max))
    return float(nu_mach_dict["nu_to_M"](nu_chosen))

def get_theta_max(mach_number):
    # returns max theta in degrees
    return prandtl_meyer_function(mach_number) / 2

def compute_R_plus(theta, nu):
    return nu - theta

def compute_R_minus(theta, nu):
    return nu + theta

def handle_corner_expansion(thetas_corner, data, IFT_dict):
    # function for handling the expansion fans emminating from point 0
    for i, theta in enumerate(thetas_corner):
        nu = theta
        R_minus = compute_R_minus(theta, nu)
        R_plus = "N/A"
        M = compute_mach_from_nu(nu_target=nu, nu_mach_dict=IFT_dict)
        mu = calculate_mach_angle(M)
        data[f"0_{i + 1}"] = {
            "R_plus": R_plus,
            "R_minus": R_minus,
            "theta": theta,
            "nu": nu,
            "M": M,
            "mu": mu,
            "theta_plus_mu": theta + mu,
            "theta_minus_mu": theta - mu,
            "x": X_START,
            "y": Y_START,
        }
    return data

def obtain_points_infront(data, current_index):
    # helper function that calculates the integer number of expansion fans in front of current expansion fan
    count = 0
    for key in data:
        if key.startswith("0_"):
            try:
                _, col = key.split("_")
                if int(col) > current_index:
                    count += 1
            except ValueError:
                continue
    return count

def enforce_symmetry(data, index):
    # Symmetry BC
    data_copy = data.copy()
    R_minus = data_copy[index]["R_minus"]
    R_plus = R_minus
    theta = 0
    nu = (R_plus + R_minus) / 2  # recompute properly
    data_copy[index]["theta"] = theta
    data_copy[index]["R_plus"] = R_plus
    data_copy[index]["nu"] = nu
    return data_copy

def calculate_nu_theta_from_riemann(R_plus, R_minus):
    # helper function for obtaining nu and theta from Riemann invariants
    nu = (R_plus + R_minus) / 2
    theta = (R_minus - R_plus) / 2
    return nu, theta

def update_nu_theta(data, index):
    # helper function that updates nu and theta in data dictionary
    data_copy = data.copy()
    R_plus = data_copy[index]["R_plus"]
    R_minus = data_copy[index]["R_minus"]
    nu, theta = calculate_nu_theta_from_riemann(R_plus, R_minus)
    data_copy[index]["theta"] = theta
    data_copy[index]["nu"] = nu
    return data_copy

def update_mach_number(data, index, IFT_dict):
    # helper function that updates Mach number in data dictionary
    data_copy = data.copy()
    nu = data_copy[index]["nu"]
    mach_number = compute_mach_from_nu(nu, IFT_dict)
    data_copy[index]["M"] = mach_number
    return data_copy

def update_mu(data, index):
    # helper function that updates Mach angle in data dictionary
    data_copy = data.copy()
    mach_number = data_copy[index]["M"]
    mu = calculate_mach_angle(mach_number)
    data_copy[index]["mu"] = mu
    return data_copy

def update_theta_pm_mu(data, index):
    # helper function that calculates and updates theta +- mu in data dictionary
    data_copy = data.copy()
    theta = data_copy[index]["theta"]
    mu = data_copy[index]["mu"]
    theta_plus_mu = theta + mu
    theta_minus_mu = theta - mu
    data_copy[index]["theta_plus_mu"] = theta_plus_mu
    data_copy[index]["theta_minus_mu"] = theta_minus_mu
    return data_copy

def calculate_xy_symmetry(x0, y0, y_center, theta_0, mu_0, theta, mu):
    # simplified version of x, y coordinates Eq, by Symmetry BC
    alpha_0P = (theta_0 - mu_0 + theta - mu) / 2
    x_p = x0 - (y0 - y_center) / np.tan(np.radians(alpha_0P))
    y_p = y_center
    return x_p, y_p

def line_intersection(xA, yA, alphaAP, xB, yB, alphaBP):
    # calculates the intersection of two lines A and B with slopes alphaAP and alphaBP
    alphaAP = np.radians(alphaAP)
    alphaBP = np.radians(alphaBP)
    xP = (xB * np.tan(alphaBP) - xA * np.tan(alphaAP) - (yB - yA)) / (np.tan(alphaBP) - np.tan(alphaAP))
    yP = yA + (xP - xA) * np.tan(alphaAP)
    return xP, yP

def get_coordinates(data, index, current_column):
    # coordinate calculator helper function (used for internal interactions)
    data_copy = data.copy()
    currentPoint = f"{index}"
    pointA = f"{int(index) - 1}"
    pointB = current_column

    def get_known_xyma(data, point):
        x = data[point]["x"]
        y = data[point]["y"]
        mu = data[point]["mu"]
        theta = data[point]["theta"]
        return x, y, mu, theta

    xA, yA, muA, thetaA = get_known_xyma(data_copy, pointA)
    xB, yB, muB, thetaB = get_known_xyma(data_copy, pointB)
    mu = data[currentPoint]["mu"]
    theta = data[currentPoint]["theta"]#
    alpha_ap = (thetaA + muA + theta + mu) / 2
    alpha_bp = (thetaB - muB + theta - mu) / 2
    xP, yP = line_intersection(xA, yA, alpha_ap, xB, yB, alpha_bp)
    return xP, yP

def extract_wall_coordinates(data):
    # helper function that computes intersection of AP and BP, simplified (and different) for wall BC
    d = data.copy()
    wall_ids = [k for k, v in d.items()
                if isinstance(k, str) and v.get("R_minus") == "N/A"]
    wall_ids.sort(key=lambda s: int(s))
    if not wall_ids:
        return d
    prev_wall_id = f"0_{NUM_EXPANSION_FANS}"
    for wid in wall_ids:
        w_idx = int(wid)
        A_id = f"{w_idx - 1}"
        B_id = prev_wall_id
        d[wid]["theta"] = float(d[A_id]["theta"])
        d[wid]["nu"] = float(d[A_id]["nu"])
        d[wid]["M"] = float(d[A_id]["M"])
        d[wid]["mu"] = float(d[A_id]["mu"])
        d[wid]["theta_plus_mu"] = d[wid]["theta"] + d[wid]["mu"]
        d[wid]["theta_minus_mu"]= d[wid]["theta"] - d[wid]["mu"]
        theta_A = d[A_id]["theta"]
        mu_A = d[A_id]["mu"]
        xA, yA  = d[A_id]["x"], d[A_id]["y"]
        theta_B = d[B_id]["theta"]
        xB, yB = d[B_id]["x"], d[B_id]["y"]
        alpha_AW = theta_A + mu_A
        alpha_BW = 0.5 * (theta_B + d[wid]["theta"])
        xW, yW = line_intersection(xA, yA, alpha_AW, xB, yB, alpha_BW)
        d[wid]["x"], d[wid]["y"] = xW, yW
        prev_wall_id = wid
    return d

def handle_points(data, IFT_dict):
    # Main logic function, that runs the main iterative MoC process, calls helper functions.
    point_up_to = 0
    data_in = data.copy()
    items = list(data_in.items())
    original_keys = list(data_in.keys())
    # print(f"_"*100)
    # print(data_in)
    for i, (key, values) in enumerate(items, 1):
        if not key.startswith("0_"):
            continue
        current_col = int(key.split("_")[1])
        points_ahead = obtain_points_infront(data_in, current_col)
        # print("_" * 50)
        # print(i)
        for j in range(1, points_ahead + 2 + 1):
            point_number = j + point_up_to
            new_key = f"{point_number}"
            data_in[new_key] = {}

            if j == 1:  # collision with centre line
                # print(f"points ahead: {point_number} | Collision with centreline")
                data_in[new_key]["R_minus"] = values["R_minus"]
                data_in = enforce_symmetry(data_in, new_key)
                data_in = update_mach_number(data_in, new_key, IFT_dict)
                data_in = update_mu(data_in, new_key)
                data_in = update_theta_pm_mu(data_in, new_key)
                x, y = calculate_xy_symmetry(X_START, Y_START, CENTRE_LINE_Y, values["theta"], values["mu"],
                                             data_in[new_key]["theta"], data_in[new_key]["mu"])
                data_in[new_key]["x"] = x
                data_in[new_key]["y"] = y
                # print(f"data_in updated to {data_in[new_key]}")
                continue

            if j == points_ahead + 2:  # collision with nozzle surface
                # print(f"points ahead: {point_number} | Collision with nozzle")
                data_in[new_key]["R_plus"] = data_in[f"{point_number - 1}"]["R_plus"]
                data_in[new_key]["R_minus"] = "N/A"  # since reflection is cancelled
                data_in[new_key]["theta"] = data_in[f"{point_number - 1}"]["theta"]
                # print(f"data_in updated to {data_in[new_key]}, theta from point {point_number - 1}")
                data_in[new_key]["nu"] = data_in[new_key]["R_plus"] + data_in[new_key]["theta"]
                data_in = update_mach_number(data_in, new_key, IFT_dict)
                if current_col == int(original_keys[-1].split("_")[1]):
                    data_in[new_key]["M"] = MACH_NUMBER  # BC
                data_in = update_mu(data_in, new_key)
                data_in = update_theta_pm_mu(data_in, new_key)
                continue

            # interior collision
            # print(f"local point: {j} | points ahead: {point_number}")
            data_in[new_key]["R_minus"] = data[f"0_{current_col + j - 1}"]["R_minus"]
            data_in[new_key]["R_plus"] = data_in[f"{point_number - 1}"]["R_plus"]
            # print(data_in[new_key])
            data_in = update_nu_theta(data_in, new_key)
            data_in = update_mach_number(data_in, new_key, IFT_dict)
            data_in = update_mu(data_in, new_key)
            data_in = update_theta_pm_mu(data_in, new_key)
            x, y = get_coordinates(data_in, f"{point_number}", f"0_{current_col + j - 1}")
            data_in[new_key]["x"] = x
            data_in[new_key]["y"] = y
            # print(f"data_in updated to {data_in[new_key]}")

        point_up_to += points_ahead + 2

    data_in = extract_wall_coordinates(data_in)
    return data_in


def print_results(data):
    # Helper function that prints data dictionary as a tabulated table in the kernel
    headers = ["Point", "R⁺", "R⁻", "θ", "ν", "M", "μ", "θ+μ", "θ−μ", "x", "y"]
    rows = []
    for point, vals in data.items():
        rows.append([
            point,
            round(vals.get("R_plus", 0), 4) if isinstance(vals.get("R_plus"), (int, float)) else vals.get("R_plus"),
            round(vals.get("R_minus", 0), 4) if isinstance(vals.get("R_minus"), (int, float)) else vals.get("R_minus"),
            round(vals.get("theta", 0), 4),
            round(vals.get("nu", 0), 4),
            round(vals.get("M", 0), 4),
            round(vals.get("mu", 0), 4),
            round(vals.get("theta_plus_mu", 0), 4),
            round(vals.get("theta_minus_mu", 0), 4),
            round(vals.get("x", 0), 4),
            round(vals.get("y", 0), 4),
        ])
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

def plot_points(data, show=True, save_path=None):
    # Helper function that plots a pretty plot to visualise data
    # NOTE: Suggest setting, in config.py, NUM_EXPANSION_FANS <= 5, for plotting clarity

    corner_ids = sorted([k for k in data if isinstance(k, str) and k.startswith("0_")],
                        key=lambda s: int(s.split("_")[1]))
    num_ids = sorted([k for k in data if k.isdigit()], key=lambda s: int(s))

    sym_ids = [k for k in num_ids
               if ("theta" in data[k]) and abs(float(data[k]["theta"])) < 1e-12
               and data[k].get("R_minus") != "N/A" and ("x" in data[k]) and ("y" in data[k])]
    wall_ids = [k for k in num_ids if data[k].get("R_minus") == "N/A" and ("x" in data[k]) and ("y" in data[k])]
    int_ids = [k for k in num_ids if (k not in sym_ids) and (k not in wall_ids)
               and ("x" in data[k]) and ("y" in data[k])]

    wall_color = "#1f77b4" # blue
    char_color = "#444444" # colour for all characteristics
    point_color = "#111111"  # dark grey for points

    cminus_label_frac = 0.565  # adjust to move labels along C- lines
    cminus_label_offset = 0.05  # perpendicular offset magnitude

    cy = float(CENTRE_LINE_Y)
    fig, ax = plt.subplots(figsize=(18, 12))

    ax.axhline(cy, color="#666666", linestyle="--", linewidth=1, label="Centreline")

    if corner_ids and wall_ids:
        throat = corner_ids[-1]
        if ("x" in data[throat]) and ("y" in data[throat]):
            wx = [data[throat]["x"]] + [data[k]["x"] for k in wall_ids]
            wy = [data[throat]["y"]] + [data[k]["y"] for k in wall_ids]
            ax.plot(wx, wy, "-", color=wall_color, linewidth=2, label="Wall")
            wy_m = [2 * cy - y for y in wy]
            ax.plot(wx, wy_m, "-", color=wall_color, linewidth=2)

            ax.scatter(wx, wy, s=40, color=wall_color, marker="s", zorder=4)
            ax.scatter(wx, wy_m, s=40, color=wall_color, marker="s", zorder=4)
            wall_label_dy = 0.04
            for wid in wall_ids:
                xw, yw = data[wid]["x"], data[wid]["y"]
                ax.text(xw, yw + wall_label_dy, wid, fontsize=8, ha="center", va="bottom",
                        color=wall_color)

    def label_point(text, x, y, dy=0.025):
        if y >= cy:
            ax.text(x, y + dy, text, fontsize=8, ha="center", va="bottom", color="#333333")

    for k in corner_ids:
        if ("x" in data[k]) and ("y" in data[k]):
            xk, yk = data[k]["x"], data[k]["y"]
            ax.scatter(xk, yk, s=22, c=point_color, marker="x", zorder=3)
            if k == "0_1":
                label_point("0", xk, yk, dy=+0.04)
            ax.scatter(xk, 2 * cy - yk, s=22, c=point_color, marker="x", zorder=3)

    for k in int_ids + sym_ids:
        xk, yk = data[k]["x"], data[k]["y"]
        ax.scatter(xk, yk, s=16, c=point_color, zorder=3)
        label_point(k, xk, yk)
        ax.scatter(xk, 2 * cy - yk, s=16, c=point_color, zorder=3)

    pair_n = min(len(corner_ids), len(sym_ids))
    for i in range(pair_n):
        k0, ks = corner_ids[i], sym_ids[i]
        if ("x" in data[k0]) and ("y" in data[k0]) and ("x" in data[ks]) and ("y" in data[ks]):
            x0, y0 = data[k0]["x"], data[k0]["y"]
            xs, ys = data[ks]["x"], data[ks]["y"]
            ax.plot([x0, xs], [y0, ys], color=char_color, linewidth=1.2, alpha=0.95)
            ax.plot([x0, xs], [2 * cy - y0, 2 * cy - ys], color=char_color, linewidth=1.2, alpha=0.95)

            t = float(np.clip(cminus_label_frac, 0.0, 1.0))
            xm = x0 + t * (xs - x0)
            ym = y0 + t * (ys - y0)
            dx, dy = xs - x0, ys - y0
            ang = np.arctan2(dy, dx)
            x_off = xm - cminus_label_offset * np.sin(ang)
            y_off = ym + cminus_label_offset * np.cos(ang)
            ax.text(x_off, y_off, k0, fontsize=8, color=char_color,
                    ha="center", va="center", rotation=np.degrees(ang), rotation_mode="anchor")

    for ks in sym_ids:
        run_keys = [ks]
        n = int(ks)
        while True:
            n += 1
            kn = str(n)
            if kn not in data or ("x" not in data[kn]) or ("y" not in data[kn]):
                break
            run_keys.append(kn)
            if data[kn].get("R_minus") == "N/A":
                break
        if len(run_keys) >= 2:
            rx = [data[k]["x"] for k in run_keys]
            ry = [data[k]["y"] for k in run_keys]
            ax.plot(rx, ry, color=char_color, linewidth=1.2, alpha=0.95)
            ry_m = [2 * cy - y for y in ry]
            ax.plot(rx, ry_m, color=char_color, linewidth=1.2, alpha=0.95)

    ax.set_xlabel(r"$\frac{x}{r}$", fontsize=16)
    ax.set_ylabel(r"$\frac{y}{r}$", fontsize=16)
    ax.set_aspect("equal")
    ax.minorticks_on()
    ax.grid(True, which="major", linewidth=0.8, alpha=0.35)
    ax.grid(True, which="minor", linewidth=0.5, alpha=0.20)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", frameon=True, fontsize=16)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    plt.close(fig)

def main():
    # main function
    global data
    if OVERWRITE:
        nu_mach_raw = create_nu_mach_dict(mach_start=1.001, mach_end=3.0, num_points=NUM_IFT_ENTRIES)
        save_as_json(nu_mach_raw, fname=IFT_JSON, overwrite=OVERWRITE)
    try:
        nu_mach_dict = load_IFT_JSON(fname=IFT_JSON)
    except FileNotFoundError:
        raise SystemExit(f"{IFT_JSON} not found. Set OVERWRITE=True to create it.")
    theta_max = get_theta_max(mach_number=MACH_NUMBER)
    theta_expansion_fans = np.linspace(EXPANSION_FANS_FROM, theta_max, num=NUM_EXPANSION_FANS)
    data = handle_corner_expansion(thetas_corner=theta_expansion_fans, data=data, IFT_dict=nu_mach_dict)
    data = handle_points(data=data, IFT_dict=nu_mach_dict)
    print_results(data)
    plot_points(data, show=True, save_path=None)

if __name__ == "__main__":
    # prevents accidental calling of main() if called from other code (NOT IMPLEMENTED YET)
    main()