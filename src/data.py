from __future__ import annotations

import itertools
import json
import os
import re
from typing import Generator, Optional

import numpy as np
import pandas as pd

from .config import DROP_BODY_PARTS


def load_train_test(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train = pd.read_csv(train_path)
    train = train.loc[
        ~(
            train["lab_id"].astype(str).str.contains("MABe22", na=False)
            & train["mouse1_condition"].astype(str).str.lower().eq("lights on")
        )
    ].copy()

    train["n_mice"] = 4 - train[
        ["mouse1_strain", "mouse2_strain", "mouse3_strain", "mouse4_strain"]
    ].isna().sum(axis=1)

    cond = (
        train["lab_id"].astype(str).str.contains("AdaptableSnail", na=False)
        & (train["frames_per_second"] == 25.0)
    )
    train = train.loc[~cond].copy()

    test = pd.read_csv(test_path)
    test["sleeping"] = (
        test["lab_id"].astype(str).str.contains("MABe22", na=False)
        & test["mouse1_condition"].astype(str).str.lower().eq("lights on")
    )
    test["n_mice"] = 4 - test[
        ["mouse1_strain", "mouse2_strain", "mouse3_strain", "mouse4_strain"]
    ].isna().sum(axis=1)

    return train, test


def build_sex_lookup(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[dict[str, dict], dict[str, dict]]:
    sex_cols = [f"mouse{i}_sex" for i in range(1, 5)]
    train_lut = (
        train[["video_id"] + sex_cols]
        .drop_duplicates("video_id")
        .set_index("video_id")
        .to_dict("index")
    )
    test_lut = (
        test[["video_id"] + sex_cols]
        .drop_duplicates("video_id")
        .set_index("video_id")
        .to_dict("index")
    )
    return train_lut, test_lut


def generate_mouse_data(
    dataset: pd.DataFrame,
    traintest: str,
    traintest_directory: Optional[str] = None,
    annotation_directory: Optional[str] = None,
    generate_single: bool = True,
    generate_pair: bool = True,
    drop_body_parts: Optional[list[str]] = None,
    verbose: bool = False,
) -> Generator:
    assert traintest in ["train", "test"]
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
    if traintest == "train" and annotation_directory is None:
        annotation_directory = traintest_directory.replace("train_tracking", "train_annotation")
    drop_body_parts = DROP_BODY_PARTS if drop_body_parts is None else drop_body_parts

    def _to_num(x):
        if isinstance(x, (int, np.integer)):
            return int(x)
        m = re.search(r"(\d+)$", str(x))
        return int(m.group(1)) if m else None

    for _, row in dataset.iterrows():
        lab_id = row.lab_id
        video_id = row.video_id
        fps = float(row.frames_per_second)
        n_mice = int(row.n_mice)
        arena_w = float(row.get("arena_width_cm", np.nan))
        arena_h = float(row.get("arena_height_cm", np.nan))
        sleeping = bool(getattr(row, "sleeping", False))
        arena_shape = row.get("arena_shape", "rectangular")

        if not isinstance(row.behaviors_labeled, str):
            continue

        track_path = os.path.join(str(traintest_directory), str(lab_id), f"{video_id}.parquet")
        vid = pd.read_parquet(track_path)
        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@drop_body_parts)")
        pvid = vid.pivot(columns=["mouse_id", "bodypart"], index="video_frame", values=["x", "y"])
        del vid
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T
        pvid = (pvid / float(row.pix_per_cm_approx)).astype("float32", copy=False)

        avail = list(pvid.columns.get_level_values("mouse_id").unique())
        avail_set = set(avail) | set(map(str, avail)) | {
            f"mouse{_to_num(a)}" for a in avail if _to_num(a) is not None
        }

        def _resolve(agent_str):
            m = re.search(r"(\d+)$", str(agent_str))
            cand = [agent_str]
            if m:
                n = int(m.group(1))
                cand = [n, n - 1, str(n), f"mouse{n}", agent_str]
            for c in cand:
                if c in avail_set:
                    if c in set(avail):
                        return c
                    for a in avail:
                        if str(a) == str(c) or f"mouse{_to_num(a)}" == str(c):
                            return a
            return None

        vb = json.loads(row.behaviors_labeled)
        vb = sorted(list({b.replace("'", "") for b in vb}))
        vb = pd.DataFrame([b.split(",") for b in vb], columns=["agent", "target", "action"])
        vb["agent"] = vb["agent"].astype(str)
        vb["target"] = vb["target"].astype(str)
        vb["action"] = vb["action"].astype(str).str.lower()

        if traintest == "train":
            try:
                annot_path = os.path.join(str(annotation_directory), str(lab_id), f"{video_id}.parquet")
                annot = pd.read_parquet(annot_path)
            except FileNotFoundError:
                continue

        def _mk_meta(index, agent_id, target_id):
            m = pd.DataFrame(
                {
                    "lab_id": lab_id,
                    "video_id": video_id,
                    "agent_id": agent_id,
                    "target_id": target_id,
                    "video_frame": index.astype("int32", copy=False),
                    "frames_per_second": np.float32(fps),
                    "sleeping": sleeping,
                    "arena_shape": arena_shape,
                    "arena_width_cm": np.float32(arena_w),
                    "arena_height_cm": np.float32(arena_h),
                    "n_mice": np.int8(n_mice),
                }
            )
            for c in ("lab_id", "video_id", "agent_id", "target_id", "arena_shape"):
                m[c] = m[c].astype("category")
            return m

        if generate_single:
            vb_single = vb.query("target == 'self'")
            for agent_str in pd.unique(vb_single["agent"]):
                col_lab = _resolve(agent_str)
                if col_lab is None:
                    if verbose:
                        print(
                            f"[skip single] {video_id} missing {agent_str} in tracking "
                            f"(avail={sorted(avail)})"
                        )
                    continue
                actions = sorted(
                    vb_single.loc[vb_single["agent"].eq(agent_str), "action"]
                    .unique()
                    .tolist()
                )
                if not actions:
                    continue

                single = pvid.loc[:, col_lab]
                meta_df = _mk_meta(single.index, agent_str, "self")

                if traintest == "train":
                    a_num = _to_num(col_lab)
                    y = pd.DataFrame(
                        False, index=single.index.astype("int32", copy=False), columns=actions
                    )
                    a_sub = annot.query("(agent_id == @a_num) & (target_id == @a_num)")
                    for i in range(len(a_sub)):
                        ar = a_sub.iloc[i]
                        a = str(ar.action).lower()
                        if a in y.columns:
                            y.loc[int(ar["start_frame"]) : int(ar["stop_frame"]), a] = True
                    yield "single", single, meta_df, y
                else:
                    yield "single", single, meta_df, actions

        if generate_pair:
            vb_pair = vb.query("target != 'self'")
            if len(vb_pair) > 0:
                allowed_pairs = set(
                    map(tuple, vb_pair[["agent", "target"]].itertuples(index=False, name=None))
                )

                for agent_num, target_num in itertools.permutations(
                    np.unique(pvid.columns.get_level_values("mouse_id")), 2
                ):
                    agent_str = f"mouse{_to_num(agent_num)}"
                    target_str = f"mouse{_to_num(target_num)}"
                    if (agent_str, target_str) not in allowed_pairs:
                        continue

                    a_col = _resolve(agent_str)
                    b_col = _resolve(target_str)
                    if a_col is None or b_col is None:
                        if verbose:
                            print(f"[skip pair] {video_id} missing {agent_str}->{target_str}")
                        continue

                    actions = sorted(
                        vb_pair.query("(agent == @agent_str) & (target == @target_str)")[
                            "action"
                        ]
                        .unique()
                        .tolist()
                    )
                    if not actions:
                        continue

                    pair_xy = pd.concat([pvid[a_col], pvid[b_col]], axis=1, keys=["A", "B"])
                    meta_df = _mk_meta(pair_xy.index, agent_str, target_str)

                    if traintest == "train":
                        a_num = _to_num(a_col)
                        b_num = _to_num(b_col)
                        y = pd.DataFrame(
                            False, index=pair_xy.index.astype("int32", copy=False), columns=actions
                        )
                        a_sub = annot.query("(agent_id == @a_num) & (target_id == @b_num)")
                        for i in range(len(a_sub)):
                            ar = a_sub.iloc[i]
                            a = str(ar.action).lower()
                            if a in y.columns:
                                y.loc[int(ar["start_frame"]) : int(ar["stop_frame"]), a] = True
                        yield "pair", pair_xy, meta_df, y
                    else:
                        yield "pair", pair_xy, meta_df, actions


__all__ = ["build_sex_lookup", "generate_mouse_data", "load_train_test"]
