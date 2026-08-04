"""LODGE (global + PDDM) dance generation wrapper."""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agentlodge.config import Settings
from agentlodge.env_paths import lodge_import_paths, use_code_paths


@dataclass
class LodgeResult:
    motion: np.ndarray
    summary: str


def _resolve_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _music_segment(fea_full: np.ndarray, gi: int, length1: int) -> np.ndarray:
    start = gi * length1
    end = (gi + 1) * length1
    if end <= fea_full.shape[0]:
        return fea_full[start:end]
    if fea_full.shape[0] >= length1:
        return fea_full[-length1:]
    reps = int(np.ceil(length1 / fea_full.shape[0]))
    return np.tile(fea_full, (reps, 1))[:length1]


def _load_modata(keymopath: str, device):
    import torch

    from dld.data.render_joints.smplfk import ax_to_6v

    if keymopath.endswith(".pkl"):
        import pickle

        pkl_data = pickle.load(open(keymopath, "rb"))
        smpl_poses = pkl_data["smpl_poses"]
        t, _ = smpl_poses.shape
        smpl_poses = smpl_poses.reshape(t, -1, 3)
        smpl_poses = torch.from_numpy(smpl_poses).to(device)
        smpl_rot = ax_to_6v(smpl_poses).reshape(t, -1)
        smpl_trans = torch.from_numpy(pkl_data["smpl_trans"]).to(device)
        modata = torch.cat([smpl_trans, smpl_rot], dim=1)
    else:
        modata = np.load(keymopath)
        if not isinstance(modata, torch.Tensor):
            modata = torch.from_numpy(modata).float()
    return modata


_LODGE_MODELS = None  # (key, model_coarse, model_fine) cached per process so a warm daemon loads once


def _lodge_models(cfg, cfg_coarse, dataset, device, get_module, torch):
    """Return the (coarse, fine) LODGE diffusion models, loading + moving them to ``device`` only the
    FIRST time in this process. torch.load of the ~1.7GB checkpoints dominates a cold run, so caching
    them is what lets a persistent warm generation daemon answer subsequent seeds in seconds."""
    global _LODGE_MODELS
    key = (str(cfg.checkpoint1), str(cfg.checkpoint2))
    if _LODGE_MODELS is not None and _LODGE_MODELS[0] == key:
        return _LODGE_MODELS[1], _LODGE_MODELS[2]
    model_coarse = get_module(cfg_coarse, dataset)
    sd = torch.load(cfg.checkpoint1, map_location="cpu", weights_only=False)["state_dict"]
    model_coarse.load_state_dict(sd, strict=True)
    model_coarse.to(device).eval()
    model_fine = get_module(cfg, dataset)
    sd = torch.load(cfg.checkpoint2, map_location="cpu", weights_only=False)["state_dict"]
    model_fine.load_state_dict(sd, strict=True)
    model_fine.to(device).eval()
    _LODGE_MODELS = (key, model_coarse, model_fine)
    return model_coarse, model_fine


def generate_lodge_dance(
    lodge_features: np.ndarray,
    settings: Settings,
    work_dir: Path,
    *,
    genre: str | None = None,
    seed: int | None = None,
) -> LodgeResult:
    """Run LODGE global + local diffusion on preprocessed librosa features.

    ``seed`` seeds torch/numpy before diffusion sampling so runs are reproducible and, across
    different seeds, produce diverse dances (used by best-of-K beat-alignment selection).
    """
    lodge_root = settings.lodge_code_path
    if not lodge_root.exists():
        raise FileNotFoundError(f"LODGE codebase not found at {lodge_root}")

    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(lodge_root)
    paths = lodge_import_paths(lodge_root)
    with use_code_paths(*paths):
        import glob
        import torch
        from omegaconf import OmegaConf

        from concat_res import concat_res
        from dld.config import parse_args
        from dld.data.FineDance_dataset import Genres_fd
        from dld.data.get_data import get_datasets
        from dld.models.get_model import get_module
        from dld.utils.logger import create_logger

        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
            np.random.seed(int(seed) % (2 ** 32 - 1))

        old_argv = sys.argv
        sys.argv = [
            "lodge",
            "--cfg",
            str(lodge_root / "configs/infer_local.yaml"),
            "--cfg_assets",
            str(lodge_root / "configs/data/assets.yaml"),
            "--soft",
            "1.0",
        ]
        try:
            cfg = parse_args(phase="demo")
        finally:
            sys.argv = old_argv
        cfg.FOLDER = str(work_dir / "lodge_experiments")
        cfg.Name = "agentlodge"
        cfg.length1 = 1024
        cfg.length2 = 256
        cfg.checkpoint1 = str(settings.lodge_global_weights_path)
        cfg.checkpoint2 = str(settings.lodge_weights_path)
        cfg.DEMO.RENDER = False
        cfg.DEMO.use_cached_features = False
        Path(cfg.FOLDER).mkdir(parents=True, exist_ok=True)

        cfg_coarse = OmegaConf.load(
            str(lodge_root / "exp/Global_Module/FineDance_Global/global_train.yaml")
        )
        cfg_coarse.DATASET = cfg.DATASET
        cfg_coarse.TRAIN.DATASETS = cfg.TRAIN.DATASETS
        cfg_coarse.TEST.DATASETS = cfg.TEST.DATASETS
        cfg_coarse.Norm = cfg.Norm
        cfg_coarse.DEBUG = cfg.DEBUG
        cfg_coarse.TRAIN.NUM_WORKERS = 0
        cfg_coarse.TEST.NUM_WORKERS = 0

        device = _resolve_device()
        logger, _ = create_logger(cfg, phase="demo")
        output_dir = (
            Path(cfg.FOLDER) / str(cfg.model.model_type) / str(cfg.NAME) / "samples"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = get_datasets(cfg, logger=logger, phase="test")[0]

        model_coarse, model_fine = _lodge_models(
            cfg, cfg_coarse, dataset, device, get_module, torch)

        music_fea_full = lodge_features.astype(np.float32)
        local_num = music_fea_full.shape[0] // cfg.length2
        music_fea_full = music_fea_full[: local_num * cfg.length2]
        if local_num == 0:
            raise ValueError("Song too short for LODGE generation")

        global_num = local_num // int(cfg.length1 / cfg.length2)
        if local_num % int(cfg.length1 / cfg.length2) != 0:
            global_num += 1
        flag = local_num % int(cfg.length1 / cfg.length2)

        song_id = "custom"
        music_fea_cat = None
        all_filenames_cat: list[str] = []
        for gi in range(global_num):
            music_fea = _music_segment(music_fea_full, gi, cfg.length1)
            music_fea = torch.from_numpy(music_fea).float().to(device).unsqueeze(0)
            name = f"{song_id}g{gi:03d}g"
            if gi == 0:
                music_fea_cat = music_fea
                all_filenames_cat = [name]
            else:
                music_fea_cat = torch.cat([music_fea_cat, music_fea], dim=0)
                all_filenames_cat.append(name)

        data_tuple = None, music_fea_cat, all_filenames_cat
        model_coarse.render_sample_ori(
            data_tuple,
            "global",
            output_dir,
            render_count=-1,
            fk_out=output_dir,
            render=False,
            setmode="normal",
            device=device,
        )

        length_fi = cfg.length2
        molist_cat: list = []
        modata13_cat: list = []
        all_localfilename_cat: list[str] = []

        for rgi in range(global_num):
            music_fea_cat = music_fea_cat.reshape(-1, length_fi, 35)
            music_fea_cat = music_fea_cat[:local_num]
            keymopath = os.path.join(
                output_dir, f"global_{rgi}_{song_id}g{str(rgi).zfill(3)}g.npy"
            )
            modata_13 = _load_modata(keymopath, device)
            modata_13_temp = (
                modata_13.clone()
                if isinstance(modata_13, torch.Tensor)
                else modata_13.copy()
            )

            if rgi > 0 and rgi < (global_num - 1):
                modata_13_temp[:8] = modata13_cat[-1][-8:]
            elif rgi > 0 and rgi == (global_num - 1):
                if flag != 0:
                    modata_13_temp = modata_13[
                        modata_13.shape[0]
                        - 8
                        - (8 * (int(cfg.length1 / cfg.length2) - 1) * flag) :
                    ]
                modata_13_temp[:8] = modata13_cat[-1][-8:]

            modata = modata_13_temp[4:-4]
            scale = 8 * (int(cfg.length1 / cfg.length2) - 1)
            molist = []
            for item in range(modata.shape[0] // scale):
                molist.append(modata[item * scale : (item + 1) * scale])
                all_localfilename_cat.append(
                    f"{song_id}g{str(rgi).zfill(3)}g_l{str(item).zfill(3)}"
                )
            modata13_cat.append(modata_13)
            molist_cat += molist

        genre_name = genre or settings.lodge_genre
        if genre_name not in Genres_fd:
            genre_name = "Hiphop"
        genre_vec = torch.from_numpy(np.array(Genres_fd[genre_name])).unsqueeze(0)

        setmode = "inpaint_soft_ddim"
        data_tuple = None, music_fea_cat, all_localfilename_cat, molist_cat
        model_fine.render_sample(
            data_tuple,
            "dod",
            output_dir,
            render_count=-1,
            fk_out=output_dir,
            render=False,
            setmode=setmode,
            cons=molist_cat,
            soft_hint="dod",
            device=device,
            genre=genre_vec,
        )

        concat_res(str(output_dir))
        concat_dir = output_dir / "concat" / "npy"
        npy_files = sorted(glob.glob(str(concat_dir / "*.npy")))
        if not npy_files:
            raise RuntimeError(f"LODGE generation produced no output in {concat_dir}")

        motion = np.load(npy_files[0]).astype(np.float32)
        summary = (
            f"LODGE two-stage pipeline (global choreography + PDDM) with genre={genre_name}; "
            f"{global_num} global segments, {local_num} local windows."
        )
        return LodgeResult(motion=motion, summary=summary)
