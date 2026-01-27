from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def safe_rolling(series, window, func, min_periods=None):
    if min_periods is None:
        min_periods = max(1, window // 4)
    return series.rolling(window, min_periods=min_periods, center=True).apply(func, raw=True)


def _scale(n_frames_at_30fps, fps, ref=30.0):
    return max(1, int(round(n_frames_at_30fps * float(fps) / ref)))


def _scale_signed(n_frames_at_30fps, fps, ref=30.0):
    if n_frames_at_30fps == 0:
        return 0
    s = 1 if n_frames_at_30fps > 0 else -1
    mag = max(1, int(round(abs(n_frames_at_30fps) * float(fps) / ref)))
    return s * mag


def _fps_from_meta(meta_df, fallback_lookup, default_fps=30.0):
    if "frames_per_second" in meta_df.columns and pd.notnull(meta_df["frames_per_second"]).any():
        return float(meta_df["frames_per_second"].iloc[0])
    vid = meta_df["video_id"].iloc[0]
    return float(fallback_lookup.get(vid, default_fps))


def _speed(cx: pd.Series, cy: pd.Series, fps: float) -> pd.Series:
    return np.hypot(cx.diff(), cy.diff()).fillna(0.0) * float(fps)


def _roll_future_mean(s: pd.Series, w: int, min_p: int = 1) -> pd.Series:
    return s.iloc[::-1].rolling(w, min_periods=min_p).mean().iloc[::-1]


def _roll_future_var(s: pd.Series, w: int, min_p: int = 2) -> pd.Series:
    return s.iloc[::-1].rolling(w, min_periods=min_p).var().iloc[::-1]


def add_curvature_features(X, center_x, center_y, fps):
    vel_x = center_x.diff()
    vel_y = center_y.diff()
    acc_x = vel_x.diff()
    acc_y = vel_y.diff()

    cross_prod = vel_x * acc_y - vel_y * acc_x
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    curvature = np.abs(cross_prod) / (vel_mag**3 + 1e-6)

    for w in [30, 60]:
        ws = _scale(w, fps)
        X[f"curv_mean_{w}"] = curvature.rolling(ws, min_periods=max(1, ws // 6)).mean()

    angle = np.arctan2(vel_y, vel_x)
    angle_change = np.abs(angle.diff())
    w = 30
    ws = _scale(w, fps)
    X[f"turn_rate_{w}"] = angle_change.rolling(ws, min_periods=max(1, ws // 6)).sum()

    return X


def add_multiscale_features(X, center_x, center_y, fps):
    speed = np.sqrt(center_x.diff() ** 2 + center_y.diff() ** 2) * float(fps)

    scales = [10, 40, 160]
    for scale in scales:
        ws = _scale(scale, fps)
        if len(speed) >= ws:
            X[f"sp_m{scale}"] = speed.rolling(ws, min_periods=max(1, ws // 4)).mean()
            X[f"sp_s{scale}"] = speed.rolling(ws, min_periods=max(1, ws // 4)).std()

    if len(scales) >= 2 and f"sp_m{scales[0]}" in X.columns and f"sp_m{scales[-1]}" in X.columns:
        X["sp_ratio"] = X[f"sp_m{scales[0]}"] / (X[f"sp_m{scales[-1]}"] + 1e-6)

    return X


def add_state_features(X, center_x, center_y, fps):
    speed = np.sqrt(center_x.diff() ** 2 + center_y.diff() ** 2) * float(fps)
    w_ma = _scale(15, fps)
    speed_ma = speed.rolling(w_ma, min_periods=max(1, w_ma // 3)).mean()

    try:
        bins = [-np.inf, 0.5 * fps, 2.0 * fps, 5.0 * fps, np.inf]
        speed_states = pd.cut(speed_ma, bins=bins, labels=[0, 1, 2, 3]).astype(float)

        for window in [60, 120]:
            ws = _scale(window, fps)
            if len(speed_states) >= ws:
                for state in [0, 1, 2, 3]:
                    X[f"s{state}_{window}"] = (
                        (speed_states == state)
                        .astype(float)
                        .rolling(ws, min_periods=max(1, ws // 6))
                        .mean()
                    )
                state_changes = (speed_states != speed_states.shift(1)).astype(float)
                X[f"trans_{window}"] = state_changes.rolling(ws, min_periods=max(1, ws // 6)).sum()
    except Exception:
        pass

    return X


def add_longrange_features(X, center_x, center_y, fps):
    for window in [120, 240]:
        ws = _scale(window, fps)
        if len(center_x) >= ws:
            X[f"x_ml{window}"] = center_x.rolling(ws, min_periods=max(5, ws // 6)).mean()
            X[f"y_ml{window}"] = center_y.rolling(ws, min_periods=max(5, ws // 6)).mean()

    for span in [60, 120]:
        s = _scale(span, fps)
        X[f"x_e{span}"] = center_x.ewm(span=s, min_periods=1).mean()
        X[f"y_e{span}"] = center_y.ewm(span=s, min_periods=1).mean()

    speed = np.sqrt(center_x.diff() ** 2 + center_y.diff() ** 2) * float(fps)
    for window in [60, 120]:
        ws = _scale(window, fps)
        if len(speed) >= ws:
            X[f"sp_pct{window}"] = speed.rolling(ws, min_periods=max(5, ws // 6)).rank(pct=True)

    return X


def add_cumulative_distance_single(X, cx, cy, fps, horizon_frames_base: int = 180, colname: str = "path_cum180"):
    L = max(1, _scale(horizon_frames_base, fps))
    step = np.hypot(cx.diff(), cy.diff())
    path = step.rolling(2 * L + 1, min_periods=max(5, L // 6), center=True).sum()
    X[colname] = path.fillna(0.0).astype(np.float32)
    return X


def add_groom_microfeatures(X, df, fps):
    parts = df.columns.get_level_values(0)
    if "body_center" not in parts or "nose" not in parts:
        return X

    cx = df["body_center"]["x"]
    cy = df["body_center"]["y"]
    nx = df["nose"]["x"]
    ny = df["nose"]["y"]

    cs = (np.sqrt(cx.diff() ** 2 + cy.diff() ** 2) * float(fps)).fillna(0)
    ns = (np.sqrt(nx.diff() ** 2 + ny.diff() ** 2) * float(fps)).fillna(0)

    w30 = _scale(30, fps)
    X["head_body_decouple"] = (ns / (cs + 1e-3)).clip(0, 10).rolling(
        w30, min_periods=max(1, w30 // 3)
    ).median()

    r = np.sqrt((nx - cx) ** 2 + (ny - cy) ** 2)
    X["nose_rad_std"] = r.rolling(w30, min_periods=max(1, w30 // 3)).std().fillna(0)

    if "tail_base" in parts:
        ang = np.arctan2(df["nose"]["y"] - df["tail_base"]["y"], df["nose"]["x"] - df["tail_base"]["x"])
        dang = np.abs(ang.diff()).fillna(0)
        X["head_orient_jitter"] = dang.rolling(
            w30, min_periods=max(1, w30 // 3)
        ).mean()

    return X


def add_interaction_features(X, mouse_pair, avail_A, avail_B, fps):
    if "body_center" not in avail_A or "body_center" not in avail_B:
        return X

    rel_x = mouse_pair["A"]["body_center"]["x"] - mouse_pair["B"]["body_center"]["x"]
    rel_y = mouse_pair["A"]["body_center"]["y"] - mouse_pair["B"]["body_center"]["y"]
    rel_dist = np.sqrt(rel_x**2 + rel_y**2)

    A_vx = mouse_pair["A"]["body_center"]["x"].diff()
    A_vy = mouse_pair["A"]["body_center"]["y"].diff()
    B_vx = mouse_pair["B"]["body_center"]["x"].diff()
    B_vy = mouse_pair["B"]["body_center"]["y"].diff()

    A_lead = (A_vx * rel_x + A_vy * rel_y) / (np.sqrt(A_vx**2 + A_vy**2) * rel_dist + 1e-6)
    B_lead = (B_vx * (-rel_x) + B_vy * (-rel_y)) / (
        np.sqrt(B_vx**2 + B_vy**2) * rel_dist + 1e-6
    )

    for window in [30, 60]:
        ws = _scale(window, fps)
        X[f"A_ld{window}"] = A_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()
        X[f"B_ld{window}"] = B_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()

    approach = -rel_dist.diff()
    chase = approach * B_lead
    w = 30
    ws = _scale(w, fps)
    X[f"chase_{w}"] = chase.rolling(ws, min_periods=max(1, ws // 6)).mean()

    for window in [60, 120]:
        ws = _scale(window, fps)
        A_sp = np.sqrt(A_vx**2 + A_vy**2)
        B_sp = np.sqrt(B_vx**2 + B_vy**2)
        X[f"sp_cor{window}"] = A_sp.rolling(ws, min_periods=max(1, ws // 6)).corr(B_sp)

    return X


def add_speed_asymmetry_future_past_single(
    X: pd.DataFrame, cx: pd.Series, cy: pd.Series, fps: float, horizon_base: int = 30, agg: str = "mean"
) -> pd.DataFrame:
    w = max(3, _scale(horizon_base, fps))
    v = _speed(cx, cy, fps)
    if agg == "median":
        v_past = v.rolling(w, min_periods=max(3, w // 4), center=False).median()
        v_fut = v.iloc[::-1].rolling(w, min_periods=max(3, w // 4)).median().iloc[::-1]
    else:
        v_past = v.rolling(w, min_periods=max(3, w // 4), center=False).mean()
        v_fut = _roll_future_mean(v, w, min_p=max(3, w // 4))
    X["spd_asym_1s"] = (v_fut - v_past).fillna(0.0)
    return X


def add_gauss_shift_speed_future_past_single(
    X: pd.DataFrame, cx: pd.Series, cy: pd.Series, fps: float, window_base: int = 30, eps: float = 1e-6
) -> pd.DataFrame:
    w = max(5, _scale(window_base, fps))
    v = _speed(cx, cy, fps)

    mu_p = v.rolling(w, min_periods=max(3, w // 4)).mean()
    va_p = v.rolling(w, min_periods=max(3, w // 4)).var().clip(lower=eps)

    mu_f = _roll_future_mean(v, w, min_p=max(3, w // 4))
    va_f = _roll_future_var(v, w, min_p=max(3, w // 4)).clip(lower=eps)

    kl_pf = 0.5 * ((va_p / va_f) + ((mu_f - mu_p) ** 2) / va_f - 1.0 + np.log(va_f / va_p))
    kl_fp = 0.5 * ((va_f / va_p) + ((mu_p - mu_f) ** 2) / va_p - 1.0 + np.log(va_p / va_f))
    X["spd_symkl_1s"] = (kl_pf + kl_fp).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def transform_single(single_mouse, body_parts_tracked, fps):
    available_body_parts = single_mouse.columns.get_level_values(0)

    X = pd.DataFrame(
        {
            f"{p1}+{p2}": np.square(single_mouse[p1] - single_mouse[p2]).sum(axis=1, skipna=False)
            for p1, p2 in itertools.combinations(body_parts_tracked, 2)
            if p1 in available_body_parts and p2 in available_body_parts
        }
    )
    X = X.reindex(
        columns=[f"{p1}+{p2}" for p1, p2 in itertools.combinations(body_parts_tracked, 2)],
        copy=False,
    )

    if all(p in single_mouse.columns for p in ["ear_left", "ear_right", "tail_base"]):
        lag = _scale(10, fps)
        shifted = single_mouse[["ear_left", "ear_right", "tail_base"]].shift(lag)
        speeds = pd.DataFrame(
            {
                "sp_lf": np.square(single_mouse["ear_left"] - shifted["ear_left"]).sum(
                    axis=1, skipna=False
                ),
                "sp_rt": np.square(single_mouse["ear_right"] - shifted["ear_right"]).sum(
                    axis=1, skipna=False
                ),
                "sp_lf2": np.square(single_mouse["ear_left"] - shifted["tail_base"]).sum(
                    axis=1, skipna=False
                ),
                "sp_rt2": np.square(single_mouse["ear_right"] - shifted["tail_base"]).sum(
                    axis=1, skipna=False
                ),
            }
        )
        X = pd.concat([X, speeds], axis=1)

    if "nose+tail_base" in X.columns and "ear_left+ear_right" in X.columns:
        X["elong"] = X["nose+tail_base"] / (X["ear_left+ear_right"] + 1e-6)

    if all(p in available_body_parts for p in ["nose", "body_center", "tail_base"]):
        v1 = single_mouse["nose"] - single_mouse["body_center"]
        v2 = single_mouse["tail_base"] - single_mouse["body_center"]
        X["body_ang"] = (v1["x"] * v2["x"] + v1["y"] * v2["y"]) / (
            np.sqrt(v1["x"] ** 2 + v1["y"] ** 2)
            * np.sqrt(v2["x"] ** 2 + v2["y"] ** 2)
            + 1e-6
        )

    if "body_center" in available_body_parts:
        cx = single_mouse["body_center"]["x"]
        cy = single_mouse["body_center"]["y"]

        for w in [5, 15, 30, 60]:
            ws = _scale(w, fps)
            roll = dict(min_periods=1, center=True)
            X[f"cx_m{w}"] = cx.rolling(ws, **roll).mean()
            X[f"cy_m{w}"] = cy.rolling(ws, **roll).mean()
            X[f"cx_s{w}"] = cx.rolling(ws, **roll).std()
            X[f"cy_s{w}"] = cy.rolling(ws, **roll).std()
            X[f"x_rng{w}"] = cx.rolling(ws, **roll).max() - cx.rolling(ws, **roll).min()
            X[f"y_rng{w}"] = cy.rolling(ws, **roll).max() - cy.rolling(ws, **roll).min()
            X[f"disp{w}"] = np.sqrt(
                cx.diff().rolling(ws, min_periods=1).sum() ** 2
                + cy.diff().rolling(ws, min_periods=1).sum() ** 2
            )
            X[f"act{w}"] = np.sqrt(
                cx.diff().rolling(ws, min_periods=1).var()
                + cy.diff().rolling(ws, min_periods=1).var()
            )

        X = add_curvature_features(X, cx, cy, fps)
        X = add_multiscale_features(X, cx, cy, fps)
        X = add_state_features(X, cx, cy, fps)
        X = add_longrange_features(X, cx, cy, fps)
        X = add_cumulative_distance_single(X, cx, cy, fps, horizon_frames_base=180)
        X = add_groom_microfeatures(X, single_mouse, fps)
        X = add_speed_asymmetry_future_past_single(X, cx, cy, fps, horizon_base=30)
        X = add_gauss_shift_speed_future_past_single(X, cx, cy, fps, window_base=30)

    if all(p in available_body_parts for p in ["nose", "tail_base"]):
        nt_dist = np.sqrt(
            (single_mouse["nose"]["x"] - single_mouse["tail_base"]["x"]) ** 2
            + (single_mouse["nose"]["y"] - single_mouse["tail_base"]["y"]) ** 2
        )
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            X[f"nt_lg{lag}"] = nt_dist.shift(l)
            X[f"nt_df{lag}"] = nt_dist - nt_dist.shift(l)

    if all(p in available_body_parts for p in ["ear_left", "ear_right"]):
        ear_xy = single_mouse[["ear_left", "ear_right"]]
        ear_d = np.sqrt(np.square(ear_xy["ear_left"] - ear_xy["ear_right"]).sum(axis=1))

        for off in (-20, -10, 10, 20):
            X[f"ear_o{off}"] = ear_d.shift(-_scale_signed(off, fps))

        w = _scale(30, fps)
        roll = ear_d.rolling(w, center=True, min_periods=1)
        X["ear_con"] = roll.std() / (roll.mean() + 1e-6)

        for w0 in (10, 30, 60):
            w = _scale(w0, fps)
            r = ear_d.rolling(w, center=True, min_periods=1)

            X[f"ear_std_{w0}"] = r.std()
            X[f"ear_range_{w0}"] = r.max() - r.min()

    return X.astype(np.float32, copy=False)


def transform_pair(mouse_pair, body_parts_tracked, fps):
    avail_A = mouse_pair["A"].columns.get_level_values(0)
    avail_B = mouse_pair["B"].columns.get_level_values(0)

    X = pd.DataFrame(
        {
            f"12+{p1}+{p2}": np.square(mouse_pair["A"][p1] - mouse_pair["B"][p2]).sum(
                axis=1, skipna=False
            )
            for p1, p2 in itertools.product(body_parts_tracked, repeat=2)
            if p1 in avail_A and p2 in avail_B
        }
    )
    X = X.reindex(
        columns=[f"12+{p1}+{p2}" for p1, p2 in itertools.product(body_parts_tracked, repeat=2)],
        copy=False,
    )

    if ("A", "ear_left") in mouse_pair.columns and ("B", "ear_left") in mouse_pair.columns:
        A_ear = mouse_pair["A"]["ear_left"]
        B_ear = mouse_pair["B"]["ear_left"]

        for w in [5, 10, 15, 30, 45, 60]:
            lag = _scale(w, fps)

            shA = A_ear.shift(lag)
            shB = B_ear.shift(lag)

            dA = A_ear - shA
            dB = B_ear - shB
            dAB = A_ear - shB

            speeds = pd.DataFrame(
                {
                    f"sp_A_{w}": np.square(dA).sum(axis=1, skipna=False),
                    f"sp_B_{w}": np.square(dB).sum(axis=1, skipna=False),
                    f"sp_AB_{w}": np.square(dAB).sum(axis=1, skipna=False),
                }
            )

            X = pd.concat([X, speeds], axis=1)

            X[f"sp_diff_AB_{w}"] = speeds[f"sp_A_{w}"] - speeds[f"sp_B_{w}"]
            X[f"sp_ratio_AB_{w}"] = speeds[f"sp_A_{w}"] / (speeds[f"sp_B_{w}"] + 1e-6)

    if "nose+tail_base" in X.columns and "ear_left+ear_right" in X.columns:
        X["elong"] = X["nose+tail_base"] / (X["ear_left+ear_right"] + 1e-6)

    if all(p in avail_A for p in ["nose", "tail_base"]) and all(p in avail_B for p in ["nose", "tail_base"]):
        dir_A = mouse_pair["A"]["nose"] - mouse_pair["A"]["tail_base"]
        dir_B = mouse_pair["B"]["nose"] - mouse_pair["B"]["tail_base"]
        X["rel_ori"] = (dir_A["x"] * dir_B["x"] + dir_A["y"] * dir_B["y"]) / (
            np.sqrt(dir_A["x"] ** 2 + dir_A["y"] ** 2)
            * np.sqrt(dir_B["x"] ** 2 + dir_B["y"] ** 2)
            + 1e-6
        )

    if all(p in avail_A for p in ["nose"]) and all(p in avail_B for p in ["nose"]):
        cur = np.square(mouse_pair["A"]["nose"] - mouse_pair["B"]["nose"]).sum(axis=1, skipna=False)
        lag = _scale(10, fps)
        shA_n = mouse_pair["A"]["nose"].shift(lag)
        shB_n = mouse_pair["B"]["nose"].shift(lag)
        past = np.square(shA_n - shB_n).sum(axis=1, skipna=False)
        X["appr"] = cur - past

    if "body_center" in avail_A and "body_center" in avail_B:
        cd = np.sqrt(
            (mouse_pair["A"]["body_center"]["x"] - mouse_pair["B"]["body_center"]["x"]) ** 2
            + (mouse_pair["A"]["body_center"]["y"] - mouse_pair["B"]["body_center"]["y"]) ** 2
        )
        X["v_cls"] = (cd < 5.0).astype(float)
        X["cls"] = ((cd >= 5.0) & (cd < 15.0)).astype(float)
        X["med"] = ((cd >= 15.0) & (cd < 30.0)).astype(float)
        X["far"] = (cd >= 30.0).astype(float)

    if "body_center" in avail_A and "body_center" in avail_B:
        A = mouse_pair["A"]["body_center"]
        B = mouse_pair["B"]["body_center"]

        cd_full = ((A - B) ** 2).sum(axis=1, skipna=False)

        dA = A.diff()
        dB = B.diff()

        coord = (dA * dB).sum(axis=1)
        As = np.sqrt((dA**2).sum(axis=1))
        Bs = np.sqrt((dB**2).sum(axis=1))

        vel_cos = (coord / (As * Bs + 1e-9)).clip(-1, 1)

        roll_cfg = dict(min_periods=1, center=True)

        def r(x, ws):
            return x.rolling(ws, **roll_cfg)

        for w in (5, 15, 30, 60):
            ws = _scale(w, fps)

            d_var = r(cd_full, ws).var()
            X[f"int{w}"] = 1 / (1 + d_var)

            X[f"co_m{w}"] = r(coord, ws).mean()
            X[f"co_s{w}"] = r(coord, ws).std()

            for tag, s in (("A", As), ("B", Bs)):
                X[f"{tag}_speed_m{w}"] = r(s, ws).mean()
                X[f"{tag}_speed_s{w}"] = r(s, ws).std()

            X[f"vel_cos_m{w}"] = r(vel_cos, ws).mean()
            X[f"vel_cos_s{w}"] = r(vel_cos, ws).std()

    if "nose" in avail_A and "nose" in avail_B:
        nxA, nyA = mouse_pair["A"]["nose"]["x"], mouse_pair["A"]["nose"]["y"]
        nxB, nyB = mouse_pair["B"]["nose"]["x"], mouse_pair["B"]["nose"]["y"]
        nn = np.sqrt((nxA - nxB) ** 2 + (nyA - nyB) ** 2)

        is_close = (nn < 10.0).astype(float)

        for lag in [10, 20, 40]:
            l = _scale(lag, fps)

            sh = nn.shift(l)
            diff = nn - sh

            X[f"nn_lg{lag}"] = sh
            X[f"nn_ch{lag}"] = diff
            X[f"cl_ps{lag}"] = is_close.rolling(l, min_periods=1).mean()

            cl_roll = is_close.rolling(l, min_periods=1)
            X[f"nn_close_run{lag}"] = cl_roll.sum()

            close_edge = is_close.diff().fillna(0)
            X[f"nn_enters{lag}"] = (close_edge == 1).rolling(l, min_periods=1).sum()
            X[f"nn_exits{lag}"] = (close_edge == -1).rolling(l, min_periods=1).sum()

            sh2 = nn.shift(2 * l)
            X[f"nn_acc{lag}"] = nn - 2 * sh + sh2

    if "nose" in avail_A and "nose" in avail_B:
        nxA, nyA = mouse_pair["A"]["nose"]["x"], mouse_pair["A"]["nose"]["y"]
        nxB, nyB = mouse_pair["B"]["nose"]["x"], mouse_pair["B"]["nose"]["y"]
        nn = np.sqrt((nxA - nxB) ** 2 + (nyA - nyB) ** 2)

        is_close = (nn < 10.0).astype(float)

        for lag in [10, 20, 40]:
            l = _scale(lag, fps)

            sh1 = nn.shift(l)
            sh2 = nn.shift(2 * l)
            diff = nn - sh1

            X[f"nn_approach_rate{lag}"] = (diff < 0).rolling(l, min_periods=1).mean()
            X[f"nn_approach_strength{lag}"] = (-diff.clip(upper=0)).rolling(
                l, min_periods=1
            ).mean()

            close_nn = nn.where(is_close == 1)

            X[f"nn_close_var{lag}"] = close_nn.rolling(l, min_periods=1).var()
            X[f"nn_close_mean{lag}"] = close_nn.rolling(l, min_periods=1).mean()

            X[f"nn_jitter{lag}"] = diff.abs().rolling(l, min_periods=1).mean()

            acc = nn - 2 * sh1 + sh2
            X[f"nn_acc_abs{lag}"] = acc.abs().rolling(l, min_periods=1).mean()

    if "body_center" in avail_A and "body_center" in avail_B:
        Avx = mouse_pair["A"]["body_center"]["x"].diff()
        Avy = mouse_pair["A"]["body_center"]["y"].diff()
        Bvx = mouse_pair["B"]["body_center"]["x"].diff()
        Bvy = mouse_pair["B"]["body_center"]["y"].diff()
        val = (Avx * Bvx + Avy * Bvy) / (
            np.sqrt(Avx**2 + Avy**2) * np.sqrt(Bvx**2 + Bvy**2) + 1e-6
        )

        for off in [-20, -10, 0, 10, 20]:
            o = _scale_signed(off, fps)
            X[f"va_{off}"] = val.shift(-o)

        w = _scale(30, fps)
        X["int_con"] = cd_full.rolling(w, min_periods=1, center=True).std() / (
            cd_full.rolling(w, min_periods=1, center=True).mean() + 1e-6
        )

        X = add_interaction_features(X, mouse_pair, avail_A, avail_B, fps)

    return X.astype(np.float32, copy=False)


__all__ = [
    "_fps_from_meta",
    "_scale",
    "_scale_signed",
    "add_curvature_features",
    "add_gauss_shift_speed_future_past_single",
    "add_groom_microfeatures",
    "add_interaction_features",
    "add_longrange_features",
    "add_multiscale_features",
    "add_speed_asymmetry_future_past_single",
    "add_state_features",
    "safe_rolling",
    "transform_pair",
    "transform_single",
]
