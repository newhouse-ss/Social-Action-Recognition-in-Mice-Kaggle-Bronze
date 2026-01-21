"""Raw port of the 'TEST-ONLY INFERENCE (LOW-RAM VERSION)' cell from the 7th-place notebook.

This is intentionally kept close to the original to preserve correctness.
Refactor targets:
- turn constants into CLI args
- separate model definition / feature building / post-processing
- add regression tests for post-processing
"""

# ====================== TEST-ONLY INFERENCE (LOW-RAM VERSION) ======================
import os, sys, json, math, gc, re, itertools, warnings, time, contextlib
from typing import List, Dict, Tuple, Set
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import concurrent.futures as _fut
import threading as _th

# ------------------------------- Config --------------------------------------
TEST_PATH   = "/kaggle/input/MABe-mouse-behavior-detection/"
TRAIN_PATH  = "/kaggle/input/MABe-mouse-behavior-detection/"

# >>> NEW: two model lists for 2×GPU
BUNDLE_PATH1 = ["/kaggle/input/mouse-v142/mbnetdual_bundle_v142.pt","/kaggle/input/mouse-v142/mbnetdual_bundle_v143.pt","/kaggle/input/mouse-v142/mbnetdual_bundle_v144.pt"]
BUNDLE_PATH2 = ["/kaggle/input/mouse-v142/mbnetdual_bundle_v149.pt","/kaggle/input/mouse-v142/mbnetdual_bundle_v150.pt","/kaggle/input/mouse-v142/mbnetdual_bundle_v151.pt"]

# Inference knobs
WINDOW_T       = 64          # model context length
STEP_T         = WINDOW_T//2
BATCH_VIDEOS   = 1           # videos per outer batch (smaller = lower RAM)
BATCH_WINDOWS  = 1024*2      # temporal windows per forward pass (GPU VRAM control)
SMOOTH_WIN     = 5           # rolling window for light smoothing
DEFAULT_THR    = 0.7
OUT_PATH       = "submission.csv"
DEBUG          = 0           # 0=off; >0: use that many rows from train.csv (non-NA behaviors)
ENSEMBLE_N     = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float32

def log(msg): 
    print(msg, flush=True)

def _gpu_mem(dev: str | None = None):
    if not torch.cuda.is_available(): 
        return "gpu=NA"
    if dev is not None:
        torch.cuda.synchronize(device=dev)
    a = torch.cuda.memory_allocated() / (1024**3)
    r = torch.cuda.memory_reserved() / (1024**3)
    return f"gpu={a:.1f}/{r:.1f} GB"

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
log(f"[boot] DEVICE={DEVICE}, torch={torch.__version__}, cuda={torch.cuda.is_available()}")

# --------------------------- Model definitions -------------------------------
class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=200000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)
    def forward(self, x):  # x: [B,T,D]
        return x + self.pe[:x.size(1)]

class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=5, d=1, p=0.2):
        super().__init__()
        pad = (k-1)//2 * d
        self.conv1 = nn.Conv1d(c_in,  c_out, k, padding=pad, dilation=d)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=pad, dilation=d)
        self.norm1 = nn.BatchNorm1d(c_out); self.norm2 = nn.BatchNorm1d(c_out)
        self.drop  = nn.Dropout(p)
        self.proj  = nn.Identity() if c_in==c_out else nn.Conv1d(c_in, c_out, 1)
    def forward(self, x):  # [B,C,T]
        r = self.proj(x)
        x = self.drop(F.gelu(self.norm1(self.conv1(x))))
        x = self.drop(F.gelu(self.norm2(self.conv2(x))))
        return x + r

class DS_TCNEncoder(nn.Module):
    def __init__(self, c_in, width=512, n_tcn=3, dropout=0.2):
        super().__init__()
        chans = [c_in, width // 2, width, width]
        dil = [1, 2, 4][:max(1, n_tcn)]
        blocks, c = [], c_in
        for i, d in enumerate(dil):
            c_out = chans[min(i + 1, len(chans) - 1)]
            blocks.append(TCNBlock(c, c_out, k=5, d=d, p=dropout))
            c = c_out
        self.net = nn.Sequential(*blocks)
        self.out_dim = c
    def forward(self, x):  # x: [B,T,C]
        x = x.transpose(1, 2)   # [B,C,T]
        x = self.net(x)
        return x.transpose(1, 2) # [B,T,C]

class CrossAttnBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.2):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff  = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model)
        )
    def forward(self, q, kv):  # [B,T,D]
        qn, kvn = self.ln_q(q), self.ln_kv(kv)
        h, _ = self.attn(qn, kvn, kvn, need_weights=False)
        x = q + self.drop(h)
        y = self.ff(self.ln2(x))
        return x + self.drop(y)

class MouseBehaviorNetDualStream(nn.Module):
    def __init__(self, in_dim: int, num_actions: int, feat_names: List[str],
                 width: int = 512, n_tcn: int = 3, nhead: int = 8, dropout: float = 0.2):
        super().__init__()
        idx_agent, idx_target, idx_rel = [], [], []
        for i, n in enumerate(map(str, feat_names)):
            if n in ("rel_dx", "rel_dy", "rel_dist"):
                idx_rel.append(i)
            elif n.startswith("t_") or n.startswith("m_t_"):
                idx_target.append(i)
            else:
                idx_agent.append(i)
        if not idx_agent or not idx_target:
            raise ValueError("Need unprefixed agent features and 't_*'/'m_t_*' target features.")
        self.register_buffer("idx_agent", torch.tensor(idx_agent, dtype=torch.long), persistent=False)
        self.register_buffer("idx_target", torch.tensor(idx_target, dtype=torch.long), persistent=False)
        self.register_buffer("idx_rel",    torch.tensor(idx_rel,    dtype=torch.long), persistent=False)

        d_agent, d_target, d_rel = len(idx_agent), len(idx_target), len(idx_rel)
        shared_c = max(d_agent, d_target)
        self.adapt_agent  = nn.Identity() if d_agent  == shared_c else nn.Linear(d_agent,  shared_c, bias=False)
        self.adapt_target = nn.Identity() if d_target == shared_c else nn.Linear(d_target, shared_c, bias=False)

        self.enc = DS_TCNEncoder(c_in=shared_c, width=width, n_tcn=n_tcn, dropout=dropout)
        self.pe  = SinusoidalPE(self.enc.out_dim)
        self.cross_at = CrossAttnBlock(self.enc.out_dim, nhead=nhead, dropout=dropout)  # agent attends target
        self.cross_ta = CrossAttnBlock(self.enc.out_dim, nhead=nhead, dropout=dropout)  # target attends agent
        self.rel_proj = nn.Linear(d_rel, self.enc.out_dim, bias=False) if d_rel > 0 else None

        fuse_in = self.enc.out_dim * 2 + (self.enc.out_dim if self.rel_proj is not None else 0)
        self.fuse = nn.Sequential(nn.LayerNorm(fuse_in), nn.Linear(fuse_in, width), nn.GELU(), nn.Dropout(dropout))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=nhead, dim_feedforward=width * 4, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True
        )
        self.refine = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, num_actions))

    def forward(self, x):  # [B,T,D]
        Xa = x.index_select(-1, self.idx_agent)
        Xt = x.index_select(-1, self.idx_target)
        Xr = x.index_select(-1, self.idx_rel) if self.idx_rel.numel() else None
        Xa = self.adapt_agent(Xa); Xt = self.adapt_target(Xt)
        Ha = self.enc(Xa); Ht = self.enc(Xt)
        Ha = self.pe(Ha); Ht = self.pe(Ht)
        A  = self.cross_at(Ha, Ht); T  = self.cross_ta(Ht, Ha)
        if Xr is not None and self.rel_proj is not None:
            R = self.rel_proj(Xr); Z = torch.cat([A, T, R], dim=-1)
        else:
            Z = torch.cat([A, T], dim=-1)
        Z = self.fuse(Z); Z = self.refine(Z)
        return self.head(Z)

# ----------------------------- Bundle I/O ------------------------------------
def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # PyTorch ≥ 2.4
    except TypeError:
        return torch.load(path, map_location="cpu")

def _infer_in_dim_from_state_dict(sd: dict) -> int:
    preferred = ["tcn.0.conv1.weight", "enc.net.0.conv1.weight"]
    for k in preferred:
        if k in sd:
            w = sd[k]; assert w.ndim in (3,4)
            return w.shape[1]
    cands = [k for k,v in sd.items() if k.endswith("conv1.weight") and getattr(v, "ndim", 0) in (3,4)]
    assert cands, "Cannot infer in_dim"
    return sd[cands[0]].shape[1]

def _infer_num_actions_from_state_dict(sd: dict) -> int:
    head_2d = [(k, v) for k, v in sd.items() if k.startswith("head.") and getattr(v, "ndim", 0) == 2]
    assert head_2d, "Cannot infer num_actions"
    last_key, last_w = sorted(head_2d, key=lambda kv: kv[0])[-1]
    return last_w.shape[0]

def load_inference_bundle(load_path: str, device: str = "cuda"):
    bundle = _safe_torch_load(load_path)
    cls_name = bundle.get("model_class", "MouseBehaviorNetDualStream")
    mk = dict(bundle.get("model_kwargs", {}))
    sd = bundle["model_state"]
    if mk.get("in_dim") is None: mk["in_dim"] = _infer_in_dim_from_state_dict(sd)
    if mk.get("num_actions") is None: mk["num_actions"] = _infer_num_actions_from_state_dict(sd)
    if cls_name == "MouseBehaviorNetDualStream" and "feat_names" not in mk:
        mk["feat_names"] = list(bundle.get("feat_cols", []))
    cls = globals().get(cls_name, None)
    assert cls is not None, f"Model class '{cls_name}' not defined"
    model = cls(**mk).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    actions   = bundle["actions"]
    feat_cols = bundle["feat_cols"]
    scaler    = bundle["scaler"]
    #thr       = bundle.get("thresholds", {})
    thr       = {'allogroom': 0.34000000000000014, 'approach': 0.9400000000000001, 'attack': 0.77, 'attemptmount': 0.53, 'avoid': 0.75, 'biteobject': 0.55, 'chase': 0.8400000000000001, 'chaseattack': 0.76, 'climb': 0.48000000000000004, 'defend': 0.59, 'dig': 0.8, 'disengage': 0.8400000000000001, 'dominance': 0.63, 'dominancegroom': 0.66, 'dominancemount': 0.93, 'ejaculate': 0.35000000000000003, 'escape': 0.8300000000000001, 'exploreobject': 0.7400000000000001, 'flinch': 0.78, 'follow': 0.89, 'freeze': 0.30000000000000004, 'genitalgroom': 0.59, 'huddle': 0.6, 'intromit': 0.49000000000000005, 'mount': 0.7500000000000001, 'rear': 0.64, 'reciprocalsniff': 0.87, 'rest': 0.66, 'run': 0.89, 'selfgroom': 0.8, 'shepherd': 0.88, 'sniff': 0.61, 'sniffbody': 0.63, 'sniffface': 0.91, 'sniffgenital': 0.68, 'submit': 0.37000000000000005, 'tussle': 0.5499999999999999}
    id_cols   = bundle.get("id_cols", ["video_id","frame","agent_id","target_id"])

    log(f"[bundle] class={cls_name} actions={len(actions)} feats={len(feat_cols)}")
    return model, actions, feat_cols, scaler, thr, id_cols

# ----------------------------- Meta & I/O ------------------------------------
def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(r"[^\w]+","_",regex=True)
                  .str.replace("__+","_",regex=True)
                  .str.strip("_"))
    return df

def read_csv_any(root: str, name: str) -> pd.DataFrame | None:
    p = os.path.join(root, name)
    if not os.path.exists(p): 
        return None
    return _norm_cols(pd.read_csv(p))

def load_meta_row_any(lab_id: str, vid: int, prefer: str = "train") -> pd.Series:
    lab_id, vid = str(lab_id).strip(), int(vid)
    order = [("train", TRAIN_PATH, "train.csv"), ("test", TEST_PATH, "test.csv")]
    if prefer != "train":
        order.reverse()
    for _, root, csv in order:
        meta = read_csv_any(root, csv)
        if meta is None: 
            continue
        hit = meta[(meta["lab_id"].astype(str) == lab_id) & (pd.to_numeric(meta["video_id"], errors="coerce") == vid)]
        if len(hit):
            return hit.iloc[0]
    raise KeyError(f"No meta row for lab_id='{lab_id}', video_id={vid}")

def tracking_path_any(lab: str, vid: int) -> str:
    lab = str(lab).strip()
    candidates = [
        os.path.join(TEST_PATH,  "test_tracking",  lab, f"{vid}.parquet"),
        os.path.join(TEST_PATH,  "train_tracking", lab, f"{vid}.parquet"),
        os.path.join(TRAIN_PATH, "test_tracking",  lab, f"{vid}.parquet"),
        os.path.join(TRAIN_PATH, "train_tracking", lab, f"{vid}.parquet"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Tracking parquet not found for lab='{lab}', vid={vid}")

# ----------------------------- Feature build ---------------------------------
DROP_PARTS = [
    'headpiece_bottombackleft','headpiece_bottombackright','headpiece_bottomfrontleft','headpiece_bottomfrontright',
    'headpiece_topbackleft','headpiece_topbackright','headpiece_topfrontleft','headpiece_topfrontright',
    'spine_1','spine_2','tail_middle_1','tail_middle_2','tail_midpoint'
]
ALL_PARTS_FULL = [
    "ear_left","ear_right","hip_left","hip_right","neck","nose","tail_base",
    "body_center","lateral_left","lateral_right","tail_tip","tail_midpoint",
    "spine_1","spine_2","tail_middle_1","tail_middle_2","head",
    "headpiece_bottombackleft","headpiece_bottombackright","headpiece_bottomfrontleft","headpiece_bottomfrontright",
    "headpiece_topbackleft","headpiece_topbackright","headpiece_topfrontleft","headpiece_topfrontright",
]
ALL_PARTS = [p for p in ALL_PARTS_FULL if p not in DROP_PARTS]

def load_tracking_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def pivot_mouse_frame(tracking_df: pd.DataFrame, parts_order: List[str]) -> Dict[int, pd.DataFrame]:
    mice = sorted(tracking_df["mouse_id"].unique())
    out = {}
    # build only needed x/y columns to keep memory in check
    for m in mice:
        sub = tracking_df.loc[tracking_df["mouse_id"]==m, ["video_frame","bodypart","x","y"]]
        wide = sub.pivot_table(index="video_frame", columns="bodypart", values=["x","y"])
        cols = [(c0, p) for p in parts_order for c0 in ("x","y")]
        wide = wide.reindex(columns=pd.MultiIndex.from_tuples(cols, names=["coord","bodypart"]))
        wide.columns = [f"{c0}_{c1}" for c0,c1 in wide.columns.to_list()]
        out[m] = wide.sort_index()
        del sub, wide
    return out

def fill_and_normalize_with_masks(wide: pd.DataFrame, pix_per_cm: float, width_pix: float, height_pix: float) -> pd.DataFrame:
    df = wide.copy()
    ever_obs = ~df.isna().all(axis=0)
    masks = pd.DataFrame({f"m_{c}": np.float32(ever_obs[c]) for c in df.columns}, index=df.index)
    df = df.interpolate(limit_direction="both").ffill().bfill().fillna(0.0)

    # FIX: don't reference 'ppcm' before assignment; use the function arg 'pix_per_cm'
    ppcm = float(pix_per_cm) if pd.notna(pix_per_cm) and float(pix_per_cm) != 0 else 1.0

    cx, cy = float(width_pix)/2.0, float(height_pix)/2.0
    x_cols = [c for c in df.columns if c.startswith("x_")]
    y_cols = [c for c in df.columns if c.startswith("y_")]
    if x_cols: df[x_cols] = (df[x_cols].astype(float) - cx) / ppcm
    if y_cols: df[y_cols] = (df[y_cols].astype(float) - cy) / ppcm
    df = df.replace([np.inf,-np.inf], np.nan).fillna(0.0)
    return pd.concat([df, masks], axis=1)


def add_velocities(wide_with_masks: pd.DataFrame, fps: float) -> pd.DataFrame:
    df = wide_with_masks.copy()
    coord_cols = [c for c in df.columns if c.startswith("x_") or c.startswith("y_")]
    vel = df[coord_cols].diff().fillna(0.0) * float(fps)
    vel.columns = [f"v{c}" for c in coord_cols]
    mask_names = [f"m_{c}" for c in coord_cols]
    have_mask = [m for m in mask_names if m in df.columns]
    mv_dict = {f"m_v{m[2:]}": df[m] for m in have_mask}
    mv = pd.DataFrame(mv_dict, index=df.index) if mv_dict else pd.DataFrame(index=df.index)
    out = pd.concat([df, vel, mv], axis=1)
    del df, vel, mv
    return out

# ---- Robust parsing for behaviors_labeled (handles funny quoting/mixed forms)
_TRIPLET_REGEX = re.compile(
    r"""^\s*
        ['"]?\s*(?P<a>(?:mouse)?\d+|self|same|agent)\s*['"]?\s*,\s*
        ['"]?\s*(?P<t>(?:mouse)?\d+|self|same|agent)\s*['"]?\s*,\s*
        ['"]?\s*(?P<act>[A-Za-z_]+)\s*['"]?
        \s*$""", re.X | re.IGNORECASE
)
def _safe_parse_behaviors_labeled(x):
    if pd.isna(x): return set()
    if isinstance(x, (list, tuple, set)): return {str(s).strip() for s in x if str(s).strip()}
    s = str(x).strip()
    if not s: return set()
    try:
        parsed = json.loads(s)
        if isinstance(parsed, (list, tuple, set)):
            return {str(t).strip() for t in parsed if str(t).strip()}
        if isinstance(parsed, str):
            st = parsed.strip()
            return {st} if st else set()
    except Exception:
        pass
    if "|" in s:
        return {t.strip() for t in s.split("|") if t.strip()}
    return {s}

def _split_triplet_smart(trip: str):
    if not isinstance(trip, str): return None
    s = trip.strip().strip("[](){} \t\r\n").replace("’", "'")
    m = _TRIPLET_REGEX.match(s)
    if m:
        return m.group("a").lower(), m.group("t").lower(), m.group("act").strip()
    parts = [p.strip().strip("\"'[](){} \t\r\n") for p in s.split(",")]
    if len(parts) < 3: return None
    return parts[0].lower(), parts[1].lower(), parts[2].strip()

def _mouse_to_int(tok: str):
    if not isinstance(tok, str): return None
    s = tok.lower().replace(" ", "").replace("mouse", "")
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else None

def _parse_triplet_to_numeric(triplet: str):
    split = _split_triplet_smart(triplet)
    if not split: return None
    a_tok, t_tok, act = split
    ai = None if a_tok in {"self","same","agent"} else _mouse_to_int(a_tok)
    if ai is None: return None
    if t_tok in {"self", "same", "agent"}:
        ti = ai
    else:
        ti = _mouse_to_int(t_tok)
        if ti is None: return None
    act = str(act).strip()
    if not act: return None
    return (ai, ti, act)

def build_active_map_string(meta_df: pd.DataFrame) -> Dict[int, Set[str]]:
    amap: Dict[int, Set[str]] = {}
    if meta_df is None or "video_id" not in meta_df.columns or "behaviors_labeled" not in meta_df.columns:
        return amap
    for vid, g in meta_df.groupby("video_id", sort=False):
        S: Set[str] = set()
        for raw in g["behaviors_labeled"].dropna():
            for trip in _safe_parse_behaviors_labeled(raw):
                parsed = _parse_triplet_to_numeric(trip)
                if parsed is None: 
                    continue
                ai, ti, act = parsed
                S.add(f"{ai},{ti},{act}")
        amap[int(vid)] = S
    return amap

# -------------------------- Pairwise features (DS) ---------------------------
def _make_target_prefixed(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if c.startswith(("x_","y_","vx_","vy_","m_x_","m_y_","m_vx_","m_vy_"))]
    mapping = {c: ("m_t_" + c[2:] if c.startswith("m_") else "t_" + c) for c in keep}
    return df[keep].rename(columns=mapping)

def pairwise_features(per_mouse_wide: Dict[int, pd.DataFrame],
                      allowed_pairs: List[Tuple[int,int]] | None = None) -> pd.DataFrame:
    if not per_mouse_wide:
        return pd.DataFrame(columns=["frame","agent_id","target_id"])
    mice = sorted(per_mouse_wide.keys())
    mice_set = set(mice)

    it = iter(mice)
    first = next(it)
    common_index = per_mouse_wide[first].index
    for m in it:
        common_index = common_index.intersection(per_mouse_wide[m].index)

    per_mouse_aligned: Dict[int, pd.DataFrame] = {}
    for m in mice:
        dfm = per_mouse_wide[m].loc[common_index]
        if len(dfm):
            per_mouse_aligned[m] = dfm

    if not per_mouse_aligned:
        return pd.DataFrame(columns=["frame","agent_id","target_id"])

    if allowed_pairs is not None:
        pairs = [(a, t) for (a, t) in allowed_pairs if a in per_mouse_aligned and t in per_mouse_aligned]
        if not pairs:
            avail = sorted(per_mouse_aligned.keys())
            pairs = list(itertools.product(avail, avail))
    else:
        avail = sorted(per_mouse_aligned.keys())
        pairs = list(itertools.product(avail, avail))

    if not pairs:
        return pd.DataFrame(columns=["frame","agent_id","target_id"])

    candidates = [p for p in ["nose","body_center","neck","tail_base","ear_left","ear_right"] if p in ALL_PARTS]

    rows = []
    for agent, target in pairs:
        A = per_mouse_aligned.get(agent)
        T = per_mouse_aligned.get(target)
        if A is None or T is None or A.empty or T.empty:
            continue

        feat = pd.concat([A, _make_target_prefixed(T)], axis=1)
        picked = None
        for p in candidates:
            Ax, Ay = f"x_{p}", f"y_{p}"
            if Ax in A and Ay in A and Ax in T and Ay in T:
                picked = p
                break

        if picked is not None:
            ax = A[f"x_{picked}"].to_numpy(); ay = A[f"y_{picked}"].to_numpy()
            tx = T[f"x_{picked}"].to_numpy(); ty = T[f"y_{picked}"].to_numpy()
            dx = tx - ax; dy = ty - ay
            feat = feat.assign(rel_dx=dx, rel_dy=dy, rel_dist=np.sqrt(dx*dx + dy*dy))
        else:
            feat = feat.assign(rel_dx=0.0, rel_dy=0.0, rel_dist=0.0)

        feat["agent_id"] = agent
        feat["target_id"] = target
        rows.append(feat)

    if not rows:
        return pd.DataFrame(columns=["frame","agent_id","target_id"])

    out = pd.concat(rows, copy=False).reset_index(names="frame")
    id_cols = ["frame","agent_id","target_id"]
    feat_cols = [c for c in out.columns if c not in id_cols]
    return out[id_cols + feat_cols]

def build_video_feats_test_only(lab_id: str, vid: int,
                                allowed_pairs: List[Tuple[int,int]] | None = None) -> pd.DataFrame:
    meta = load_meta_row_any(lab_id, vid, prefer="train")
    fps = float(meta.get("frames_per_second", 30.0))
    ppcm = meta.get("pix_per_cm_approx", np.nan)
    ppcm = float(ppcm) if pd.notna(ppcm) and float(ppcm) != 0 else np.nan
    width_pix  = float(meta.get("video_width_pix", np.nan))
    height_pix = float(meta.get("video_height_pix", np.nan))
    if (not pd.notna(ppcm)) and all(pd.notna(meta.get(c, np.nan)) for c in ["arena_width_cm","arena_height_cm"]):
        aw = float(meta["arena_width_cm"]); ah = float(meta["arena_height_cm"])
        if aw>0 and ah>0 and pd.notna(width_pix) and pd.notna(height_pix):
            ppcm = ((width_pix/aw) + (height_pix/ah)) / 2.0
    if not pd.notna(ppcm) or ppcm == 0: ppcm = 1.0

    path = tracking_path_any(lab_id, vid)
    track = load_tracking_parquet(path)
    track = track[~track["bodypart"].isin(DROP_PARTS)].copy()

    per_mouse = pivot_mouse_frame(track, ALL_PARTS)
    del track
    for m in list(per_mouse.keys()):
        pm = fill_and_normalize_with_masks(per_mouse[m], ppcm, width_pix, height_pix)
        per_mouse[m] = add_velocities(pm, fps)
        del pm

    if allowed_pairs is not None:
        mice_present = set(per_mouse.keys())
        allowed_pairs = [(a, t) for (a, t) in allowed_pairs if a in mice_present and t in mice_present]
        if not allowed_pairs:
            allowed_pairs = None  # fall back to all x all

    feats = pairwise_features(per_mouse, allowed_pairs=allowed_pairs)
    del per_mouse
    feats["video_id"] = int(vid)
    return feats

# ------------------------------- Masking -------------------------------------
def mask_probs_numpy_rle(probs: pd.DataFrame, ACTIONS: List[str], active_map: Dict[int, Set[str]],
                         copy=True) -> pd.DataFrame:
    df = probs.copy() if copy else probs
    if not len(df): return df
    action_cols = [f"action_{a}" for a in ACTIONS]
    act_block = df[action_cols].to_numpy(copy=False)
    N, A = act_block.shape
    vid = df["video_id"].to_numpy(np.int64, copy=False)
    ag  = df["agent_id"].to_numpy(np.int64, copy=False)
    tg  = df["target_id"].to_numpy(np.int64, copy=False)

    act_pos = {a: i for i, a in enumerate(ACTIONS)}
    allow: Dict[int, Dict[Tuple[int,int], np.ndarray]] = {}
    for v, triples in active_map.items():
        v = int(v)
        d = allow.setdefault(v, {})
        for s in triples:
            sag, stg, sa = s.split(",")
            key = (int(sag), int(stg))
            arr = d.get(key)
            if arr is None:
                arr = np.zeros(A, dtype=bool)
                d[key] = arr
            i = act_pos.get(sa)
            if i is not None:
                arr[i] = True

    if N == 1:
        starts = np.array([0], dtype=np.int64); ends = np.array([1], dtype=np.int64)
    else:
        change = (vid[1:] != vid[:-1]) | (ag[1:] != ag[:-1]) | (tg[1:] != tg[:-1])
        boundaries = np.flatnonzero(change) + 1
        starts = np.concatenate(([0], boundaries))
        ends   = np.concatenate((boundaries, [N]))

    for s, e in zip(starts, ends):
        v, a_, t_ = int(vid[s]), int(ag[s]), int(tg[s])
        d = allow.get(v)
        if d is None:
            act_block[s:e, :] = 0.0; continue
        mask = d.get((a_, t_))
        if mask is None:
            act_block[s:e, :] = 0.0; continue
        if not mask.all():
            disallowed = ~mask
            act_block[s:e, disallowed] = 0.0

    df[action_cols] = act_block
    return df

# ----------------------------- Intervals -------------------------------------
from typing import List, Dict
import numpy as np
import pandas as pd

def probs_to_nonoverlapping_intervals(
    prob_df: pd.DataFrame,
    actions: List[str],
    min_len: int = 3,
    max_gap: int = 2,
    lab: str | None = None,
    tie_config: Dict[str, dict] | None = None,
) -> pd.DataFrame:
    """
    Convert frame-level probs to non-overlapping intervals for a *single lab*.

    - Thresholds are taken from TT_PER_LAB_NN[lab] and TT_PER_LAB_XGB[lab].
    - If `tie_config` is given and contains `lab`, we apply per-lab tie manipulation
      on frames where multiple actions pass threshold.

    tie_config[lab] format (per what you've been using in validation):
      {
        "boost":    { "action_name": delta, ... },
        "penalize": { "action_name": delta, ... },
        "prefer":   [
            ("winner_action", "loser_action", margin),
            ...
        ],
      }
    """
    out: list[dict] = []
    act_cols = [f"action_{a}" for a in actions]

    # Per-lab thresholds (you’re overriding per_action_thresh here anyway)
    per_action_thresh = TT_PER_LAB_NN[lab]
    print(f"Using NN lab = {lab} with th = {per_action_thresh}")
    per_action_thresh2 = TT_PER_LAB_XGB[lab]
    print(f"Using XGB lab = {lab} with th = {per_action_thresh2}")

    thr2 = np.array(
        [per_action_thresh2.get(a, 0.2) if per_action_thresh2 else 0.2 for a in actions],
        dtype=np.float32,
    )
    thr = np.array(
        [per_action_thresh.get(a, 0.75) if per_action_thresh else 0.75 for a in actions],
        dtype=np.float32,
    )

    # Optional per-lab tie rules
    lab_tie_cfg = None
    if tie_config is not None and lab is not None and lab in tie_config:
        lab_tie_cfg = tie_config[lab]
        # map action name -> column index
        action_to_idx = {a: i for i, a in enumerate(actions)}
    else:
        action_to_idx = {}

    for (vid, ag, tg), grp in prob_df.groupby(["video_id", "agent_id", "target_id"], sort=False):
        g = grp.sort_values("frame")
        frames = g["frame"].to_numpy()
        P = g[act_cols].to_numpy(np.float32)  # [T, num_actions]

        # blend of NN + XGB thresholds
        blended_thr = (1.0 - XGB_WGT) * thr[None, :] + XGB_WGT * thr2[None, :]
        pass_mask = (P >= blended_thr)

        # ---- Tie manipulation (per lab) ----
        P_adj = P.copy()
        if lab_tie_cfg is not None:
            # frames where 2+ actions pass threshold
            multi_mask = (pass_mask.sum(axis=1) > 1)

            if multi_mask.any():
                # 1) boosts
                boost_cfg = lab_tie_cfg.get("boost", {})
                for act, delta in boost_cfg.items():
                    idx = action_to_idx.get(act, None)
                    if idx is not None:
                        P_adj[multi_mask, idx] += float(delta)

                # 2) penalties
                penalize_cfg = lab_tie_cfg.get("penalize", {})
                for act, delta in penalize_cfg.items():
                    idx = action_to_idx.get(act, None)
                    if idx is not None:
                        P_adj[multi_mask, idx] -= float(delta)

                # 3) explicit preferences: ("winner", "loser", margin)
                prefer_cfg = lab_tie_cfg.get("prefer", [])
                for winner_act, loser_act, margin in prefer_cfg:
                    wi = action_to_idx.get(winner_act, None)
                    li = action_to_idx.get(loser_act, None)
                    if wi is None or li is None:
                        continue
                    # Only when both pass threshold on that frame
                    fm = multi_mask & pass_mask[:, wi] & pass_mask[:, li]
                    if fm.any():
                        # Nudge winner upwards on these ambiguous frames
                        P_adj[fm, wi] += float(margin)

                # keep probabilities in a sane range
                np.clip(P_adj, 0.0, 1.0, out=P_adj)

        # --- Best-label decoding with thresholds + adjusted scores ---
        P_masked = np.where(pass_mask, P_adj, -np.inf)
        best_idx = np.argmax(P_masked, axis=1)
        best_val = P_masked[np.arange(len(P_masked)), best_idx]
        label = np.where(np.isfinite(best_val), best_idx, -1)

        # --- Fill short gaps up to max_gap (unchanged) ---
        if max_gap > 0:
            i = 0
            while i < len(label):
                if label[i] >= 0:
                    j = i
                    while j + 1 < len(label) and label[j + 1] == label[i]:
                        j += 1
                    k = j + 1
                    while k < len(label) and label[k] == -1:
                        k += 1
                    if k < len(label) and label[k] == label[i] and (k - j - 1) <= max_gap:
                        label[j + 1:k] = label[i]
                        j = k
                    i = j + 1
                else:
                    i += 1

        # --- Convert label sequence to intervals (unchanged) ---
        def flush(s, e, idx):
            if s is None:
                return
            if frames[e] - frames[s] + 1 >= min_len:
                out.append(
                    {
                        "video_id": int(vid),
                        "agent_id": int(ag),
                        "target_id": int(tg),
                        "action": actions[idx],
                        "start_frame": int(frames[s]),
                        "stop_frame": int(frames[e] + 1),
                    }
                )

        s = None
        cur = -1
        for i_t, idx in enumerate(label):
            if idx != cur:
                if cur >= 0:
                    flush(s, i_t - 1, cur)
                s = i_t if idx >= 0 else None
                cur = idx
        if cur >= 0:
            flush(s, len(label) - 1, cur)

    return pd.DataFrame(out)

# ------------------------------- Utils ---------------------------------------
def ensure_feat_alignment(df: pd.DataFrame, id_cols: List[str], FEAT_COLS: List[str]) -> pd.DataFrame:
    for c in FEAT_COLS:
        if c not in df.columns:
            df[c] = 0.0
    out = df[id_cols + FEAT_COLS].copy()
    for c in ("video_id","frame","agent_id","target_id"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(np.int64)
    for c in FEAT_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(np.float32)
    return out

def robust_scale_inplace(df: pd.DataFrame, FEAT_COLS: List[str], scaler: dict):
    med = pd.Series(scaler.get("med", {}), dtype=np.float32)
    iqr = pd.Series(scaler.get("iqr", {}), dtype=np.float32).replace(0, 1.0)
    low = float(scaler.get("clip_low", -5.0)); high = float(scaler.get("clip_high", 5.0))
    X = df[FEAT_COLS].astype(np.float32)
    X = (X - med.reindex(FEAT_COLS).fillna(0.0).astype(np.float32)) / iqr.reindex(FEAT_COLS).fillna(1.0).astype(np.float32)
    X = X.clip(low, high).replace([np.inf,-np.inf], np.nan).fillna(0.0).astype(np.float32)
    df.loc[:, FEAT_COLS] = X.values

@torch.inference_mode()
def predict_groups(model: nn.Module, actions: List[str], df_part: pd.DataFrame,
                   T=WINDOW_T, step=STEP_T, batch_windows=BATCH_WINDOWS, device: str = None) -> pd.DataFrame:
    """Stream groups to reduce RAM; stack up to `batch_windows` temporal windows per pass.
       NEW: optional `device` to run on specific GPU in threaded mode."""
    use_device = device if device is not None else DEVICE
    id_cols = ["video_id","frame","agent_id","target_id"]
    feat_cols = [c for c in df_part.columns if c not in id_cols]
    act_cols = [f"action_{a}" for a in actions]

    gobj = df_part.groupby(["video_id","agent_id","target_id"], sort=False)
    n_groups = gobj.ngroups
    log(f"[predict] groups={n_groups} | T={T} step={step} stack={batch_windows} | { _gpu_mem(use_device) }")

    out_parts = []
    gi = 0
    for (vid,a,t), grp in gobj:
        gi += 1
        g = grp.sort_values("frame")
        X = g[feat_cols].to_numpy(np.float32, copy=False)
        F = len(g)

        preds  = np.zeros((F, len(actions)), np.float32)
        counts = np.zeros((F, 1), np.float32)

        starts = list(range(0, F, step))
        wi = 0
        while wi < len(starts):
            this = starts[wi:wi+batch_windows]
            if not this: break

            lens = []
            maxT = 0
            for s in this:
                e = min(s + T, F)
                lens.append((s,e))
                maxT = max(maxT, e - s)

            batch = np.empty((len(lens), maxT, X.shape[1]), np.float32)
            for i,(s,e) in enumerate(lens):
                L = e - s
                batch[i, :L] = X[s:e]
                if L < maxT:
                    batch[i, L:maxT] = X[e-1:e]

            tb = torch.from_numpy(batch).to(use_device, non_blocking=True)
            ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if torch.cuda.is_available() else contextlib.nullcontext()
            with ctx:
                out = model(tb).sigmoid().detach().cpu().numpy().astype(np.float32, copy=False)

            for i,(s,e) in enumerate(lens):
                L = e - s
                preds[s:e]  += out[i, :L]
                counts[s:e] += 1.0

            del batch, tb, out, lens
            wi += batch_windows

        preds /= np.maximum(counts, 1.0)
        part = pd.DataFrame(preds, columns=act_cols)
        part["video_id"]=int(vid); part["agent_id"]=int(a); part["target_id"]=int(t)
        part["frame"] = g["frame"].to_numpy(copy=False)
        out_parts.append(part)

        if gi == 1 or gi == n_groups or (gi % 25 == 0):
            log(f"[predict] {gi}/{n_groups} (vid={vid}, a={a}, t={t}) | { _gpu_mem(use_device) }")
        del g, X, preds, counts, part, grp
        if gi % 200 == 0:
            gc.collect()

    res = pd.concat(out_parts, ignore_index=True, copy=False) if out_parts else df_part.head(0)
    del out_parts
    gc.collect()
    return res

# ---------------------- NEW: multi-model helpers (2×GPU) ---------------------
class _DeviceEnsemble:
    def __init__(self, device: str, bundle_paths: List[str]):
        self.device = device
        self.paths = [p for p in (bundle_paths or []) if isinstance(p, str)]
        self.models: List[nn.Module] = []
        self.scalers: List[dict] = []
        self.thresholds_merged: Dict[str, float] = {}
        self.first_thresholds: Dict[str, float] | None = None   # <<< NEW
        self.ACTIONS: List[str] = []
        self.FEAT_COLS: List[str] = []
        self.id_cols: List[str] = []
        self.lock = _th.Lock()

    def load(self):
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            idx = int(self.device.split(":")[1])
            torch.cuda.set_device(idx)
        for i, p in enumerate(self.paths):
            m, A, F, sc, thr, ids = load_inference_bundle(p, device=self.device)
            self.models.append(m)
            self.scalers.append(sc)
            if i == 0:
                self.ACTIONS = A; self.FEAT_COLS = F; self.id_cols = ids
                if thr:                        # <<< NEW: remember thresholds from the first model on this device
                    self.first_thresholds = dict(thr)
            else:
                if A != self.ACTIONS:
                    log(f"[warn] ACTIONS mismatch on {p}; using first bundle's order")
                if F != self.FEAT_COLS:
                    log(f"[warn] FEAT_COLS mismatch on {p}; columns will be aligned by name")
                if ids != self.id_cols:
                    log(f"[warn] id_cols mismatch on {p}; using first bundle's")
            if thr:
                self.thresholds_merged.update(thr)

    def predict_avg(self, feats: pd.DataFrame, T=WINDOW_T, step=STEP_T, batch_windows=BATCH_WINDOWS) -> pd.DataFrame:
        """Average logits/probs across this device's models."""
        if not self.models:
            return pd.DataFrame(columns=["video_id","agent_id","target_id","frame"] + [f"action_{a}" for a in self.ACTIONS])
        parts = []
        for mi, model in enumerate(self.models, 1):
            out = predict_groups(model, self.ACTIONS, feats, T=T, step=step, batch_windows=batch_windows, device=self.device)
            parts.append(out)
            gc.collect()
        if len(parts) == 1:
            return parts[0]
        # align on keys then mean action columns
        key = ["video_id","agent_id","target_id","frame"]
        merged = parts[0]
        for k in range(1, len(parts)):
            merged = merged.merge(parts[k], on=key, how="inner", suffixes=(None, f"__m{k}"))
        act_cols = [f"action_{a}" for a in self.ACTIONS]
        # collect same-named cols + any suffixed copies
        all_act_cols = [c for c in merged.columns if c.startswith("action_")]
        # average per action across model outputs
        for a in act_cols:
            like = [c for c in all_act_cols if c.split("__")[0] == a]
            merged[a] = merged[like].mean(axis=1).astype(np.float32)
        # keep only one copy
        keep = key + act_cols
        return merged[keep]

def _weighted_average_two_probs(probs1: pd.DataFrame | None, n1: int,
                                probs2: pd.DataFrame | None, n2: int,
                                ACTIONS: List[str]) -> pd.DataFrame:
    """Average two prob tables with possibly different presence; weights = counts of models."""
    key = ["video_id","agent_id","target_id","frame"]
    act_cols = [f"action_{a}" for a in ACTIONS]
    if (probs1 is None or probs1.empty) and (probs2 is None or probs2.empty):
        return pd.DataFrame(columns=key + act_cols)
    if probs1 is None or probs1.empty:
        return probs2.copy()
    if probs2 is None or probs2.empty:
        return probs1.copy()
    m = probs1.merge(probs2, on=key, how="inner", suffixes=("_a","_b"))
    for a in act_cols:
        m[a] = ((m[f"{a}_a"] * float(max(n1,0))) + (m[f"{a}_b"] * float(max(n2,0)))) / float(max(n1,0) + max(n2,0))
    return m[key + act_cols].astype({a: np.float32 for a in act_cols})

def smooth_probs_inplace(probs: pd.DataFrame, actions: List[str], win: int = SMOOTH_WIN):
    if win is None or win <= 1 or probs.empty: 
        return
    act_cols = [f"action_{a}" for a in actions]
    probs.sort_values(["video_id","agent_id","target_id","frame"], inplace=True)
    pad = win // 2
    kernel = np.ones(win, dtype=np.float32) / win
    gobj = probs.groupby(["video_id","agent_id","target_id"], sort=False)
    for _, idx in gobj.groups.items():
        idx = np.asarray(idx)
        arr = probs.loc[idx, act_cols].to_numpy(np.float32, copy=False)
        for j in range(arr.shape[1]):
            v = arr[:, j]
            if v.size:
                vp = np.pad(v, (pad, pad), mode="edge")
                arr[:, j] = np.convolve(vp, kernel, mode="valid").astype(np.float32, copy=False)
        probs.loc[idx, act_cols] = arr

# --------------------------------- Main --------------------------------------
def main():
    t0_all = time.time()

    # Load meta/test list (same as before)
    if DEBUG and DEBUG > 0:
        train_meta = read_csv_any(TRAIN_PATH, "train.csv")
        if train_meta is None:
            raise FileNotFoundError("train.csv not found for DEBUG mode")
        train_meta = train_meta.loc[~train_meta["behaviors_labeled"].isna()].reset_index()
        rows = train_meta.index.to_list()[:DEBUG]
        sel = train_meta.iloc[:DEBUG].copy()
        TEST_LIST = list(sel.drop_duplicates(subset=["lab_id","video_id"], keep="first")[["lab_id","video_id"]]
                         .itertuples(index=False, name=None))
        active_map_str = build_active_map_string(sel)
        log(f"[DEBUG] Using {len(sel)} rows from train.csv (behaviors_labeled notna). Row indices: {rows}")
        log(f"[DEBUG] Unique (lab, video) count: {len(TEST_LIST)}")
        prefer_meta = "train"
        del train_meta, sel
    else:
        test_meta = read_csv_any(TEST_PATH, "test.csv")
        if test_meta is None:
            raise FileNotFoundError("test.csv not found")
        TEST_LIST = list(test_meta.drop_duplicates(subset=["lab_id","video_id"], keep="first")[["lab_id","video_id"]]
                         .itertuples(index=False, name=None))
        active_map_str = build_active_map_string(test_meta)
        prefer_meta = "train"
        del test_meta

    # Preflight for tracking/meta availability
    GOOD, BAD = [], []
    for lab, vid in TEST_LIST:
        try:
            _ = tracking_path_any(lab, vid)
            _ = load_meta_row_any(lab, vid, prefer=prefer_meta)
            GOOD.append((lab, vid))
        except Exception as e:
            BAD.append((lab, vid, str(e)))
    if BAD:
        log(f"[preflight] skipping {len(BAD)} videos due to missing meta/parquet")
    TEST_LIST = GOOD if GOOD else TEST_LIST
    del GOOD, BAD
    gc.collect()

    # Prepare output
    pd.DataFrame(columns=["row_id","video_id","agent_id","target_id","action","start_frame","stop_frame"]).to_csv(OUT_PATH, index=False)

    # >>> NEW: build per-device ensembles once (models pinned to each GPU)
    ngpu = torch.cuda.device_count()
    dev0 = "cuda:0" if torch.cuda.is_available() and ngpu >= 1 else "cpu"
    dev1 = "cuda:1" if torch.cuda.is_available() and ngpu >= 2 else None

    ens0 = _DeviceEnsemble(dev0, BUNDLE_PATH1); ens0.load()
    ACTIONS = ens0.ACTIONS; FEAT_COLS = ens0.FEAT_COLS; id_cols = ens0.id_cols

    ens1 = None
    if dev1 is not None and BUNDLE_PATH2:
        ens1 = _DeviceEnsemble(dev1, BUNDLE_PATH2); ens1.load()
        if not ACTIONS:  ACTIONS  = ens1.ACTIONS
        if not FEAT_COLS: FEAT_COLS = ens1.FEAT_COLS
        if not id_cols:   id_cols   = ens1.id_cols

    if not ACTIONS or not FEAT_COLS or not id_cols:
        raise RuntimeError("Could not infer ACTIONS/FEAT_COLS/id_cols from bundles")

    # >>> NEW: thresholds from the FIRST available model only (ens0's first, else ens1's first)
    if ens0.first_thresholds is not None:
        thresholds = dict(ens0.first_thresholds)
        thresholds_source = "BUNDLE_PATH1[0]"
    elif ens1 is not None and ens1.first_thresholds is not None:
        thresholds = dict(ens1.first_thresholds)
        thresholds_source = "BUNDLE_PATH2[0]"
    else:
        thresholds = {}
        thresholds_source = "default (none in bundles)"
    log(f"[bundle] thresholds source: {thresholds_source}")

    # Process in multi-video batches
    log(f"[infer] videos 1-{len(TEST_LIST)} / {len(TEST_LIST)} (batch={BATCH_VIDEOS}, win_batch={BATCH_WINDOWS}) | { _gpu_mem() }")
    row_counter = 0
    total_pred_rows = 0

    for bi in range(0, len(TEST_LIST), BATCH_VIDEOS):
        t_batch = time.time()
        batch = TEST_LIST[bi:bi+BATCH_VIDEOS]
        tag = f"[batch {bi//BATCH_VIDEOS + 1}/{(len(TEST_LIST)+BATCH_VIDEOS-1)//BATCH_VIDEOS}]"
        labs_vids_str = " ".join([f"({lab},{vid})" for lab,vid in batch])
        log(f"{tag} building feats for {len(batch)} vids: {labs_vids_str}")

        # ---------- Single-thread: build feats for this batch ----------
        feats_list = []
        for lab, vid in batch:
            allowed_set = active_map_str.get(int(vid), set())
            pairs = sorted({(int(s.split(",")[0]), int(s.split(",")[1])) for s in allowed_set}) if allowed_set else None
            f = build_video_feats_test_only(lab, vid, allowed_pairs=pairs)
            feats_list.append(f)
            log(f"{tag} built feats vid={vid} rows={len(f)}")
        feats = pd.concat(feats_list, ignore_index=True, copy=False)
        del feats_list
        gc.collect()

        total_pred_rows += len(feats)
        log(f"{tag} concat feats rows={len(feats)} | { _gpu_mem() }")

        # Align, scale (use ens0's scaler if available, else ens1's)
        feats = ensure_feat_alignment(feats, id_cols, FEAT_COLS)
        base_scaler = ens0.scalers[0] if ens0.scalers else (ens1.scalers[0] if ens1 and ens1.scalers else None)
        if base_scaler is None:
            raise RuntimeError("No scaler found in any bundle")
        robust_scale_inplace(feats, FEAT_COLS, base_scaler)

        # ---------- Two threads: run per-device ensemble inference ----------
        def _run_on_ens(ens: _DeviceEnsemble):
            if ens is None or not ens.models:
                return None
            return ens.predict_avg(feats, T=WINDOW_T, step=STEP_T, batch_windows=BATCH_WINDOWS)

        with _fut.ThreadPoolExecutor(max_workers=2) as ex:
            futs = []
            futs.append(ex.submit(_run_on_ens, ens0))
            if ens1 is not None:
                futs.append(ex.submit(_run_on_ens, ens1))
            results = [f.result() for f in futs]

        probs0 = results[0] if len(results) >= 1 else None
        probs1 = results[1] if len(results) >= 2 else None
        n0 = len(ens0.models)
        n1 = len(ens1.models) if (ens1 is not None) else 0

        # Weighted combine across devices
        probs = _weighted_average_two_probs(probs0, n0, probs1, n1, ACTIONS)
        log(f"{tag} predict done | probs_rows={len(probs)} | { _gpu_mem() }")
        del probs0, probs1
        gc.collect()

       #############################
        # -------- Ensemble with per-video probs from model1..modelN --------
        key_cols = ["video_id", "agent_id", "target_id", "frame"]

        # Ensure key dtypes are consistent (usually ints) in base probs
        for c in key_cols:
            probs[c] = pd.to_numeric(probs[c], errors="coerce").astype("Int64")

        # Collect the corresponding probs for all videos in this batch
        vids_in_batch = probs["video_id"].dropna().unique().tolist()

        # Start with base model (current probs) as the running ensemble
        probs_ens = probs.copy()
        CC = [c for c in probs_ens.columns if "action" in c]
        probs_ens[CC] = 0.0
        n_models = 0  # current ensemble has 0 models; base probs are zeroed out

        # Loop over external models model1, model2, ..., model{ENSEMBLE_N}
        for k in range(1, ENSEMBLE_N + 1):
            model_dir = f"/tmp/model{k}"

            # Load all per-video parquet files for this batch for model k
            model_dfs = []
            for vid in vids_in_batch:
                vid_int = int(vid)
                path = os.path.join(model_dir, f"vid{vid_int}_probs.parquet")
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Ensemble model folder '{model_dir}' probs file not found "
                        f"for video_id={vid_int}: {path}"
                    )
                df_m = pd.read_parquet(path)
                # Make sure keys are same dtype
                for c in key_cols:
                    df_m[c] = pd.to_numeric(df_m[c], errors="coerce").astype("Int64")
                model_dfs.append(df_m)

            if not model_dfs:
                raise RuntimeError(
                    f"No probs loaded from '{model_dir}' for batch videos: {vids_in_batch}"
                )

            probs_m = pd.concat(model_dfs, ignore_index=True)
            del model_dfs
            gc.collect()

            # Merge current ensemble with this model's probs on exact key
            # Keep existing columns unchanged, suffix the new model's overlapping columns
            probs_merged = probs_ens.merge(
                probs_m,
                on=key_cols,
                how="inner",
                suffixes=("", f"_m{k}"),
            )
            del probs_m
            gc.collect()

            if len(probs_merged) != len(probs_ens):
                log(
                    f"{tag} WARNING: after merging model{k}, rows {len(probs_merged)} "
                    f"!= previous {len(probs_ens)} "
                    f"(some rows may be missing in model{k} files)"
                )

            # Update running equal-weight average for each action column
            # new_mean = (old_mean * n_models + new_model) / (n_models + 1)
            for a in ACTIONS:
                base = f"action_{a}"
                col_new = f"{base}_m{k}"

                if base not in probs_merged.columns or col_new not in probs_merged.columns:
                    raise KeyError(
                        f"Missing columns for action '{a}' when merging model{k}: "
                        f"{base} or {col_new} not found in merged probs"
                    )

                probs_merged[base] = (
                    probs_merged[base] * n_models + probs_merged[col_new]
                ) / (n_models + 1)

            # Drop the temporary per-model columns for this k
            drop_cols_k = [f"action_{a}_m{k}" for a in ACTIONS]
            probs_merged = probs_merged.drop(columns=drop_cols_k)

            # Replace ensemble with updated merged/averaged version
            probs_ens = probs_merged
            n_models += 1

            gc.collect()

        # At this point probs_ens has equal-weighted average over (base + ENSEMBLE_N models)
        if len(probs_ens) != len(probs):
            log(
                f"{tag} WARNING: final merged rows {len(probs_ens)} != original {len(probs)} "
                f"(some rows may be missing in one or more ensemble model files)"
            )

        probs = probs_ens
        del probs_ens
        gc.collect()
        #############################

        V = probs.video_id.unique()
        probs['lab_id'] = probs['video_id'].map(V2L).fillna("unknown")
        labb = probs.lab_id.values[0]

        ### INSERT FILTER HERE ###
        lab_key = str(labb)
        allowed_actions = TRAIN_LAB_ACTIONS.get(lab_key)

        if allowed_actions is not None and len(probs):
            # Columns for actions that are allowed for THIS lab according to TRAIN
            allowed_action_cols = {f"action_{a}" for a in allowed_actions}

            # All action_* columns present
            all_action_cols = [c for c in probs.columns if c.startswith("action_")]

            # Disallowed according to train: zero them out so they can't win argmax
            disallowed_cols = [c for c in all_action_cols if c not in allowed_action_cols]
            if disallowed_cols:
                probs.loc[:, disallowed_cols] = 0.0
                log(f"{tag} lab={lab_key}: zeroed {len(disallowed_cols)} train-disallowed actions")
        else:
            log(f"{tag} lab={lab_key}: no TRAIN_LAB_ACTIONS entry (no extra filter)")
        ### END FILTER ###

        ### BEGIN FILTER ###
        CC = [f"action_{a}" for a in SELF_ACTIONS]
        probs.loc[probs.agent_id != probs.target_id, CC] = 0.0

        CC = [f"action_{a}" for a in PAIR_ACTIONS]
        probs.loc[probs.agent_id == probs.target_id, CC] = 0.0
        ### END FILTER ###
        
        try:
            files = [f"/tmp/xgb_preds/p{k}.pqt" for k in V]
            xgb = pd.read_parquet(files)
            probs2 = probs.merge(xgb,on=['video_id','agent_id','target_id','frame'],how='left').fillna(0)
            RMV = []
            for a in ACTIONS:
                n0 = f"action_{a}"
                n1 = f"action_{a}_x"
                n2 = f"action_{a}_y"
                probs2[n0] = (1-XGB_WGT)*probs2[n1] + XGB_WGT*probs2[n2]
                RMV.append(n1)
                RMV.append(n2)
            probs2 = probs2.drop(RMV,axis=1)
            probs2 = probs2[COLS]
        except:
            log(f"[XGB PREDS] no preds for {V}")
            probs2 = probs.copy()
            for a in ACTIONS:
                n0 = f"action_{a}"
                probs2[n0] = (1-XGB_WGT)*probs2[n0] + XGB_WGT*0.2

        # Smooth 
        smooth_probs_inplace(probs2, ACTIONS, win=SMOOTH_WIN)
        log(f"{tag} smooth done")

        # Mask by active set
        if active_map_str and len(probs2):
            probs2 = mask_probs_numpy_rle(probs2, ACTIONS, active_map_str, copy=False)
            log(f"{tag} mask done")

        # Intervals
        thr_map = thresholds if thresholds else {a: DEFAULT_THR for a in ACTIONS}
        sub = probs_to_nonoverlapping_intervals(probs2, ACTIONS, min_len=0, 
                                                max_gap=7, lab=labb, tie_config=TIE_CONFIG_V2)
        del probs, probs2
        gc.collect()

        wrote = 0
        if len(sub):
            sub = sub[["video_id","agent_id","target_id","action","start_frame","stop_frame"]].copy()
            sub = sub.pipe(lambda df: df.assign(
                video_id=pd.to_numeric(df["video_id"], errors="coerce").fillna(-1).astype(int),
                start_frame=pd.to_numeric(df["start_frame"], errors="coerce").fillna(0).astype(int),
                stop_frame=pd.to_numeric(df["stop_frame"], errors="coerce").fillna(0).astype(int),
                agent_id=pd.to_numeric(df["agent_id"], errors="coerce").fillna(0).astype(int),
                target_id=pd.to_numeric(df["target_id"], errors="coerce").fillna(0).astype(int),
                action=df["action"].astype(str),
            ))
            sub = sub[sub["stop_frame"] >= sub["start_frame"]]
            sub = sub[sub["action"].isin(ACTIONS)]
            # schema clean (writes "self" when agent==target)
            def _format_agent_mouse(n: int) -> str: return f"mouse{int(n)}"
            def _format_target_mouse(ai: int, ti: int) -> str: return "self" if int(ai)==int(ti) else f"mouse{int(ti)}"
            sub["agent_id"]  = sub["agent_id"].apply(_format_agent_mouse)
            sub["target_id"] = [ _format_target_mouse(ai, ti) for ai,ti in zip(sub["agent_id"].str.replace("mouse","").astype(int),
                                                                               sub["target_id"].astype(int)) ]
            sub = sub[["video_id","agent_id","target_id","action","start_frame","stop_frame"]].drop_duplicates()
            if len(sub):
                sub = sub.reset_index(drop=True)
                sub.insert(0, "row_id", range(row_counter, row_counter + len(sub)))
                row_counter += len(sub)
                sub.to_csv(OUT_PATH, mode="a", header=False, index=False)
                wrote = len(sub)
        del sub
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        dt = time.time() - t_batch
        log(f"{tag} DONE in {dt:.1f}s | wrote_rows={wrote} | preds_total_rows_so_far={total_pred_rows} | { _gpu_mem() }")

    dt_all = time.time() - t0_all
    log(f"[done] wrote {OUT_PATH} | total_pred_rows={total_pred_rows} | total_time={dt_all:.1f}s | { _gpu_mem() }")

if __name__ == "__main__":
    # Optional: improves conv/BN autotuning for fixed shapes
    torch.backends.cudnn.benchmark = True
    main()
# ==================== END TEST-ONLY INFERENCE (LOW-RAM VERSION) ====================
