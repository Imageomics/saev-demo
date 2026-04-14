"""Streamlit app for SAE feature visualization over local images.

This version loads an SAE checkpoint from a run directory, computes sparse
features directly for all images in a demo image folder, and visualizes
top-activating images per selected feature.
"""

import dataclasses
import importlib
import json
import pathlib
import typing as tp
from collections.abc import Callable

import beartype
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

import saev.data.models
import saev.data.shards
import saev.nn


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class LoadedModels:
    run_dpath: pathlib.Path
    n_patches: int
    d_model: int
    d_sae: int
    preprocess: Callable[[Image.Image], torch.Tensor]
    resize: Callable[[Image.Image], Image.Image]
    recorder: saev.data.shards.RecordedTransformer
    sae: saev.nn.SparseAutoencoder


@beartype.beartype
def get_streamlit():
    return importlib.import_module("streamlit")


@beartype.beartype
def resolve_run_paths(run_dpath: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Resolve config and checkpoint paths from a run directory."""
    cfg_fpath = run_dpath / "checkpoint" / "config.json"
    ckpt_fpath = run_dpath / "checkpoint" / "sae.pt"
    if cfg_fpath.is_file() and ckpt_fpath.is_file():
        return cfg_fpath, ckpt_fpath

    cfg_fpath = run_dpath / "config.json"
    ckpt_fpath = run_dpath / "sae.pt"
    if cfg_fpath.is_file() and ckpt_fpath.is_file():
        return cfg_fpath, ckpt_fpath

    msg = (
        "Could not find run files. Expected either "
        "<run>/checkpoint/{config.json,sae.pt} or <run>/{config.json,sae.pt}."
    )
    raise FileNotFoundError(msg)


@beartype.beartype
def load_models_from_run(run_dpath: pathlib.Path, device: str) -> LoadedModels:
    cfg_fpath, ckpt_fpath = resolve_run_paths(run_dpath)

    with open(cfg_fpath) as fd:
        cfg = tp.cast(dict[str, tp.Any], json.load(fd))

    train_data = tp.cast(dict[str, tp.Any], cfg.get("train_data", {}))
    shards_dpath = pathlib.Path(tp.cast(str, train_data.get("shards", "")))
    msg = (
        "Run config is missing train_data.shards, which is needed to recover "
        "backbone model metadata."
    )
    assert str(shards_dpath) != "", msg
    msg = f"Shards directory from run config does not exist: '{shards_dpath}'."
    assert shards_dpath.is_dir(), msg

    layer = int(train_data.get("layer", -2))
    md = saev.data.shards.Metadata.load(shards_dpath)
    msg = f"Layer {layer} is not available in metadata layers {md.layers}."
    assert layer in md.layers, msg

    model_cls = saev.data.models.load_model_cls(md.family)
    preprocess, _sample_transform = model_cls.make_transforms(
        md.ckpt, md.content_tokens_per_example
    )
    resize = model_cls.make_resize(md.ckpt, md.content_tokens_per_example, scale=1.0)

    make_model = tp.cast(tp.Callable[[str], tp.Any], model_cls)
    backbone = make_model(md.ckpt).to(device).eval()
    recorder = saev.data.shards.RecordedTransformer(
        model=backbone,
        content_tokens_per_example=md.content_tokens_per_example,
        cls_token=True,
        layers=(layer,),
    ).to(device)
    recorder.eval()

    sae = saev.nn.load(ckpt_fpath, device=device).to(device).eval()

    return LoadedModels(
        run_dpath=run_dpath,
        n_patches=md.content_tokens_per_example,
        d_model=md.d_model,
        d_sae=sae.cfg.d_sae,
        preprocess=preprocess,
        resize=resize,
        recorder=recorder,
        sae=sae,
    )


@beartype.beartype
def list_demo_images(imgs_dpath: pathlib.Path) -> list[pathlib.Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    paths = sorted(
        p for p in imgs_dpath.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    return paths


@beartype.beartype
def make_overlay(img: Image.Image, acts_p: np.ndarray) -> Image.Image:
    img_arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h_px, w_px, _ = img_arr.shape
    n_patches = acts_p.size
    grid = int(np.sqrt(n_patches))
    msg = f"n_patches={n_patches} is not square."
    assert grid * grid == n_patches, msg

    heat = acts_p.reshape(grid, grid)
    max_v = float(np.max(heat))
    if max_v <= 0:
        return Image.fromarray(img_arr.astype(np.uint8))

    heat = heat / max_v
    patch_h = h_px // grid
    patch_w = w_px // grid
    heat_up = np.kron(heat, np.ones((patch_h, patch_w), dtype=np.float32))
    heat_up = heat_up[:h_px, :w_px]

    heat_rgb = colormaps["inferno"](heat_up)[..., :3]
    alpha = np.clip(0.8 * heat_up, 0.0, 0.8)[..., None]
    out = (1.0 - alpha) * (img_arr / 255.0) + alpha * heat_rgb
    out = np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out)


@beartype.beartype
def compute_feature_maps(
    models: LoadedModels,
    image_paths: list[pathlib.Path],
    feature_i: list[int],
    device: str,
    batch_size: int,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, Image.Image], dict[int, str]]:
    feat_tensor = torch.tensor(feature_i, device=device, dtype=torch.int64)

    maps_by_example: dict[int, np.ndarray] = {}
    img_by_example: dict[int, Image.Image] = {}
    path_by_example: dict[int, str] = {}

    n_examples = len(image_paths)
    if progress_cb is not None:
        progress_cb(0, n_examples)

    start_i = 0
    while start_i < n_examples:
        end_i = min(start_i + batch_size, n_examples)
        chunk_paths = image_paths[start_i:end_i]

        xs = []
        for ex_i, fpath in enumerate(chunk_paths, start=start_i):
            img = Image.open(fpath).convert("RGB")
            xs.append(models.preprocess(img))
            img_by_example[ex_i] = models.resize(img)
            path_by_example[ex_i] = str(fpath)

        x_bchw = torch.stack(xs, dim=0).to(device)
        with torch.inference_mode():
            _ignored, rec_bltc = models.recorder(x_bchw)
            acts_bpd = rec_bltc[:, 0, 1:, :].to(device)
            msg = (
                f"Patch activations shape {tuple(acts_bpd.shape)} "
                f"mismatch with n_patches={models.n_patches}."
            )
            assert acts_bpd.shape[1] == models.n_patches, msg

            enc = models.sae.encode(acts_bpd.reshape(-1, models.d_model)).f_x
            enc_bps = enc.reshape(len(chunk_paths), models.n_patches, models.d_sae)
            enc_bpf = enc_bps[:, :, feat_tensor].cpu().numpy()

        for local_i, ex_i in enumerate(range(start_i, end_i)):
            maps_by_example[ex_i] = enc_bpf[local_i].T

        start_i = end_i
        if progress_cb is not None:
            progress_cb(end_i, n_examples)

    return maps_by_example, img_by_example, path_by_example


@beartype.beartype
def get_top_examples_from_maps(
    maps_by_example: dict[int, np.ndarray],
    *,
    feature_i: int,
    feat_pos: dict[int, int],
    n_top: int,
) -> list[tuple[int, float]]:
    slot_i = feat_pos[feature_i]
    scored = [
        (ex_i, float(np.max(feat_maps_fp[slot_i])))
        for ex_i, feat_maps_fp in maps_by_example.items()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [(ex_i, score) for ex_i, score in scored if score > 0]
    return top[:n_top]


def run_app() -> None:
    st = get_streamlit()
    st.set_page_config(layout="wide", page_title="SAE Feature Activations")
    st.markdown(
        """
        <style>
        [data-testid="stImage"] img {
            border-radius: 0 !important;
            object-fit: contain !important;
            height: auto !important;
            max-height: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("SAE Feature Activations")
    st.caption(
        "Load an SAE run checkpoint, compute sparse features for images in demo_imgs, and browse top-activating images per feature."
    )

    default_run = "./saev/runs/inat"
    default_imgs = str((pathlib.Path(__file__).resolve().parent / "demo_imgs"))

    run_path = st.sidebar.text_input("Run directory", value=default_run)
    imgs_path = st.sidebar.text_input("Demo images folder", value=default_imgs)
    feature_start = int(
        st.sidebar.number_input("Feature start index", min_value=0, value=0)
    )
    feature_end = int(
        st.sidebar.number_input("Feature end index (inclusive)", min_value=0, value=50)
    )
    n_top = int(
        st.sidebar.number_input(
            "Top images per feature", min_value=1, max_value=50, value=5
        )
    )
    batch_size = int(
        st.sidebar.number_input(
            "Batch size", min_value=1, max_value=512, value=32
        )
    )

    auto_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = st.sidebar.selectbox(
        "Device", options=[auto_device, "cpu", "cuda"], index=0
    )
    if device == "cuda" and not torch.cuda.is_available():
        st.warning("CUDA requested but unavailable, falling back to CPU.")
        device = "cpu"

    if not st.sidebar.button("Generate visualization", type="primary"):
        st.info("Set options in the sidebar, then click Generate visualization.")
        return

    run_dpath = pathlib.Path(run_path).expanduser().resolve()
    imgs_dpath = pathlib.Path(imgs_path).expanduser().resolve()
    if not run_dpath.is_dir():
        st.error(f"Run directory does not exist: {run_dpath}")
        return
    if not imgs_dpath.is_dir():
        st.error(f"Demo images folder does not exist: {imgs_dpath}")
        return

    image_paths = list_demo_images(imgs_dpath)
    if len(image_paths) == 0:
        st.error(f"No images found in {imgs_dpath}")
        return

    with st.spinner("Loading SAE and backbone models from run..."):
        models = load_models_from_run(run_dpath, device)

    if feature_end < feature_start:
        feature_start, feature_end = feature_end, feature_start
    feature_start = min(feature_start, models.d_sae - 1)
    feature_end = min(feature_end, models.d_sae - 1)
    features = list(range(feature_start, feature_end + 1))

    if not features:
        st.error("Feature range is empty after bounds checks.")
        return

    st.write(
        f"Computing features [{feature_start}, {feature_end}] "
        f"({len(features)} total) for {len(image_paths)} images."
    )

    with st.spinner("Generating sparse feature maps..."):
        progress = st.progress(0.0, text=f"Computing activations: 0/{len(image_paths)}")

        def on_progress(done: int, total: int) -> None:
            if total <= 0:
                progress.progress(1.0, text="Computing activations: 0/0")
                return
            frac = float(done) / float(total)
            progress.progress(frac, text=f"Computing activations: {done}/{total}")

        maps_by_example, img_by_example, path_by_example = compute_feature_maps(
            models,
            image_paths=image_paths,
            feature_i=features,
            device=device,
            batch_size=batch_size,
            progress_cb=on_progress,
        )

    feat_pos = {feat_i: i for i, feat_i in enumerate(features)}
    top_examples_by_feature: dict[int, list[tuple[int, float]]] = {}
    for feat_i in features:
        top_examples_by_feature[feat_i] = get_top_examples_from_maps(
            maps_by_example,
            feature_i=feat_i,
            feat_pos=feat_pos,
            n_top=n_top,
        )

    tabs = st.tabs([f"feature {feat_i}" for feat_i in features])
    for tab, feat_i in zip(tabs, features, strict=True):
        with tab:
            top_examples = top_examples_by_feature[feat_i]
            if not top_examples:
                st.warning(f"Feature {feat_i} has no non-zero activations.")
                continue

            st.write(f"Top {len(top_examples)} images for feature {feat_i}")
            cols = st.columns(5)
            for i, (ex_i, sparse_max) in enumerate(top_examples):
                col = cols[i % 5]
                with col:
                    feat_maps_fp = maps_by_example[ex_i]
                    acts_p = feat_maps_fp[feat_pos[feat_i]]
                    overlay = make_overlay(img_by_example[ex_i], acts_p)
                    st.image(
                        overlay,
                        caption=(
                            f"idx={ex_i} | sparse_max={sparse_max:.4g} | "
                            f"patch_max={float(np.max(acts_p)):.4g}"
                        ),
                        use_container_width=True,
                    )
                    st.caption(path_by_example[ex_i])


if __name__ == "__main__":
    run_app()
