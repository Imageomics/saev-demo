"""Streamlit app for SAE feature visualization.

By default this visualizes features 0..19, shows the top-10 maximally activating images
per feature, and overlays patch-level sparse activation maps.
"""

import dataclasses
import importlib
import json
import pathlib
import typing as tp
from collections.abc import Callable

import beartype
import numpy as np
import scipy.sparse
import torch
from matplotlib import colormaps
from PIL import Image

import saev.data
import saev.data.datasets
import saev.data.shards
import saev.disk
import saev.nn


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class RunArtifacts:
    run_dpath: pathlib.Path
    shard_id: str
    inference_dpath: pathlib.Path
    shards_dpath: pathlib.Path
    layer: int
    n_patches: int
    d_model: int
    d_sae: int
    dataset: saev.data.datasets.ImgFolderDataset
    resize: Callable[[Image.Image], Image.Image]
    recorder: saev.data.shards.RecordedTransformer | None
    sae: saev.nn.SparseAutoencoder | None
    token_acts_ts: scipy.sparse.csc_array | None
    mean_values_s: np.ndarray | None
    sparsity_s: np.ndarray | None


@beartype.beartype
def get_streamlit():
    return importlib.import_module("streamlit")


@beartype.beartype
def safe_load_pt(path: pathlib.Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


@beartype.beartype
def infer_shard_id(inference_dpath: pathlib.Path) -> str:
    shard_ids = sorted([
        child.name for child in inference_dpath.iterdir() if child.is_dir()
    ])
    msg = (
        f"Expected exactly one shard under '{inference_dpath}', found {len(shard_ids)}."
    )
    assert len(shard_ids) == 1, msg
    return shard_ids[0]


@beartype.beartype
def load_run_artifacts(
    run_dpath: pathlib.Path,
    shard_id: str | None,
    inference_shard_dpath: pathlib.Path | None,
    dataset_root_dpath: pathlib.Path | None,
    device: str,
    recompute: bool,
    load_sparse_artifacts: bool,
) -> RunArtifacts:
    run = saev.disk.Run(run_dpath)
    if inference_shard_dpath is None:
        inference_root_dpath = run.inference
        shard_id = shard_id or infer_shard_id(inference_root_dpath)
        inference_dpath = inference_root_dpath / shard_id
    else:
        inference_dpath = inference_shard_dpath
        shard_id = inference_dpath.name
    msg = f"Inference directory does not exist: '{inference_dpath}'."
    assert inference_dpath.is_dir(), msg

    with open(inference_dpath / "config.json") as fd:
        inference_cfg = json.load(fd)

    shards_dpath = pathlib.Path(
        inference_cfg["data"]["shards"]
    )  # same params as inference
    md = saev.data.Metadata.load(shards_dpath)
    layer = int(inference_cfg["data"]["layer"])
    msg = f"Layer {layer} missing from metadata layers {md.layers}."
    assert layer in md.layers, msg

    data_cfg = md.make_data_cfg()
    if dataset_root_dpath is not None:
        msg = f"Dataset root does not exist: '{dataset_root_dpath}'."
        assert dataset_root_dpath.is_dir(), msg
        data_cfg = saev.data.datasets.ImgFolder(root=dataset_root_dpath)
    msg = f"Expected ImgFolder dataset config, got {type(data_cfg)}."
    assert isinstance(data_cfg, saev.data.datasets.ImgFolder), msg

    model_cls = saev.data.models.load_model_cls(md.family)
    if recompute:
        data_transform, _ = model_cls.make_transforms(
            md.ckpt, md.content_tokens_per_example
        )
        dataset = saev.data.datasets.get_dataset(
            data_cfg, data_transform=data_transform
        )
    else:
        dataset = saev.data.datasets.get_dataset(data_cfg)
    msg = f"Expected ImgFolderDataset, got {type(dataset)}."
    assert isinstance(dataset, saev.data.datasets.ImgFolderDataset), msg

    resize = model_cls.make_resize(md.ckpt, md.content_tokens_per_example, scale=1.0)
    recorder: saev.data.shards.RecordedTransformer | None = None
    sae: saev.nn.SparseAutoencoder | None = None
    if recompute:
        make_model = tp.cast(tp.Callable[[str], tp.Any], model_cls)
        model = make_model(md.ckpt).to(device).eval()
        recorder = saev.data.shards.RecordedTransformer(
            model=model,
            content_tokens_per_example=md.content_tokens_per_example,
            cls_token=True,
            layers=(layer,),
        ).to(device)
        recorder.eval()
        sae = saev.nn.load(run.ckpt, device=device).to(device).eval()

    token_acts_ts: scipy.sparse.csc_array | None = None
    mean_values_s: np.ndarray | None = None
    sparsity_s: np.ndarray | None = None
    d_sae: int
    if load_sparse_artifacts:
        token_acts_ts = scipy.sparse.load_npz(
            inference_dpath / "token_acts.npz"
        ).tocsc()
        mean_values_s = safe_load_pt(inference_dpath / "mean_values.pt").cpu().numpy()
        sparsity_s = safe_load_pt(inference_dpath / "sparsity.pt").cpu().numpy()

        n_tokens, d_sae = token_acts_ts.shape
        msg = f"n_tokens={n_tokens} is not divisible by content tokens {md.content_tokens_per_example}."
        assert n_tokens % md.content_tokens_per_example == 0, msg
        msg = f"mean_values shape {mean_values_s.shape} != ({d_sae},)."
        assert mean_values_s.shape == (d_sae,), msg
        msg = f"sparsity shape {sparsity_s.shape} != ({d_sae},)."
        assert sparsity_s.shape == (d_sae,), msg
    else:
        run_cfg = tp.cast(dict[str, tp.Any], run.config)
        sae_cfg = tp.cast(dict[str, tp.Any], run_cfg["sae"])
        d_sae = int(sae_cfg["d_sae"])

    return RunArtifacts(
        run_dpath=run_dpath,
        shard_id=shard_id,
        inference_dpath=inference_dpath,
        shards_dpath=shards_dpath,
        layer=layer,
        n_patches=md.content_tokens_per_example,
        d_model=md.d_model,
        d_sae=d_sae,
        dataset=dataset,
        resize=resize,
        recorder=recorder,
        sae=sae,
        token_acts_ts=token_acts_ts,
        mean_values_s=mean_values_s,
        sparsity_s=sparsity_s,
    )


@beartype.beartype
def get_topk_index_fpath(inference_dpath: pathlib.Path, *, k: int) -> pathlib.Path:
    return inference_dpath / f"top_examples_k{k}.npz"


@beartype.beartype
def compute_topk_examples(
    token_acts_ts: scipy.sparse.csc_array,
    *,
    n_patches: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    _n_tokens, d_sae = token_acts_ts.shape
    ex_i_sf = np.full((d_sae, k), -1, dtype=np.int64)
    scores_sf = np.full((d_sae, k), np.nan, dtype=np.float32)

    for feat_i in range(d_sae):
        start_i = int(token_acts_ts.indptr[feat_i])
        end_i = int(token_acts_ts.indptr[feat_i + 1])
        token_i = token_acts_ts.indices[start_i:end_i]
        vals = token_acts_ts.data[start_i:end_i]
        if vals.size == 0:
            continue

        order_i = np.argsort(vals)[::-1]
        seen: set[int] = set()
        write_i = 0
        for row_i in order_i.tolist():
            ex_i = int(token_i[row_i] // n_patches)
            if ex_i in seen:
                continue
            seen.add(ex_i)
            ex_i_sf[feat_i, write_i] = ex_i
            scores_sf[feat_i, write_i] = float(vals[row_i])
            write_i += 1
            if write_i >= k:
                break

    return ex_i_sf, scores_sf


@beartype.beartype
def load_or_make_topk_examples(
    inference_dpath: pathlib.Path,
    token_acts_ts: scipy.sparse.csc_array,
    *,
    n_patches: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    topk_fpath = get_topk_index_fpath(inference_dpath, k=k)
    if topk_fpath.is_file():
        topk = np.load(topk_fpath)
        ex_i_sf = topk["ex_i_sf"]
        scores_sf = topk["scores_sf"]
        d_sae = token_acts_ts.shape[1]
        msg = f"top-k ex index shape {ex_i_sf.shape} != ({d_sae}, {k})."
        assert ex_i_sf.shape == (d_sae, k), msg
        msg = f"top-k score index shape {scores_sf.shape} != ({d_sae}, {k})."
        assert scores_sf.shape == (d_sae, k), msg
        return ex_i_sf, scores_sf

    ex_i_sf, scores_sf = compute_topk_examples(token_acts_ts, n_patches=n_patches, k=k)
    np.savez_compressed(topk_fpath, ex_i_sf=ex_i_sf, scores_sf=scores_sf)
    return ex_i_sf, scores_sf


@beartype.beartype
def get_top_examples_for_feature(
    token_acts_ts: scipy.sparse.csc_array,
    *,
    feature_i: int,
    n_patches: int,
    n_top: int,
    topk_ex_i_sf: np.ndarray | None = None,
    topk_scores_sf: np.ndarray | None = None,
) -> list[tuple[int, float]]:
    if (
        topk_ex_i_sf is not None
        and topk_scores_sf is not None
        and n_top <= topk_ex_i_sf.shape[1]
    ):
        ex_i_f = topk_ex_i_sf[feature_i, :n_top]
        score_f = topk_scores_sf[feature_i, :n_top]
        top_examples: list[tuple[int, float]] = []
        for ex_i, score in zip(ex_i_f.tolist(), score_f.tolist(), strict=True):
            if ex_i < 0 or np.isnan(score):
                continue
            top_examples.append((int(ex_i), float(score)))
        return top_examples

    start_i = int(token_acts_ts.indptr[feature_i])
    end_i = int(token_acts_ts.indptr[feature_i + 1])
    token_i = token_acts_ts.indices[start_i:end_i]
    vals = token_acts_ts.data[start_i:end_i]
    if vals.size == 0:
        return []

    order_i = np.argsort(vals)[::-1]
    seen = set()
    top_examples: list[tuple[int, float]] = []
    for row_i in order_i.tolist():
        ex_i = int(token_i[row_i] // n_patches)
        if ex_i in seen:
            continue
        seen.add(ex_i)
        top_examples.append((ex_i, float(vals[row_i])))
        if len(top_examples) >= n_top:
            break
    return top_examples


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
def recompute_maps(
    artifacts: RunArtifacts,
    *,
    example_i: list[int],
    feature_i: list[int],
    device: str,
    batch_size: int,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, Image.Image], dict[int, str]]:
    dataset = artifacts.dataset
    n_patches = artifacts.n_patches
    msg = "Recompute requested but recorder is missing."
    assert artifacts.recorder is not None, msg
    msg = "Recompute requested but SAE is missing."
    assert artifacts.sae is not None, msg
    recorder = artifacts.recorder
    sae = artifacts.sae
    feat_tensor = torch.tensor(feature_i, device=device, dtype=torch.int64)

    maps_by_example: dict[int, np.ndarray] = {}
    img_by_example: dict[int, Image.Image] = {}
    path_by_example: dict[int, str] = {}

    n_examples = len(example_i)
    if progress_cb is not None:
        progress_cb(0, n_examples)
    start_i = 0
    while start_i < n_examples:
        end_i = min(start_i + batch_size, n_examples)
        chunk_i = example_i[start_i:end_i]

        xs = []
        for ex_i in chunk_i:
            sample = dataset[ex_i]
            x = sample["data"]
            msg = f"Expected tensor for sample['data'], got {type(x)}."
            assert isinstance(x, torch.Tensor), msg
            xs.append(x)

            path, _target = dataset.samples[ex_i]
            path_by_example[ex_i] = path
            img = Image.open(path).convert("RGB")
            img_by_example[ex_i] = artifacts.resize(img)

        x_bchw = torch.stack(xs, dim=0).to(device)

        with torch.inference_mode():
            _ignored, rec_bltc = recorder(x_bchw)
            # content tokens only, matching inference data.tokens="content"
            acts_bpd = rec_bltc[:, 0, 1:, :].to(device)
            msg = f"Patch activations shape {tuple(acts_bpd.shape)} mismatch with n_patches={n_patches}."
            assert acts_bpd.shape[1] == n_patches, msg
            enc = sae.encode(acts_bpd.reshape(-1, artifacts.d_model)).f_x
            enc_bps = enc.reshape(len(chunk_i), n_patches, artifacts.d_sae)
            enc_bpf = enc_bps[:, :, feat_tensor].cpu().numpy()

        for local_i, ex_i in enumerate(chunk_i):
            # shape: (n_features, n_patches)
            maps_by_example[ex_i] = enc_bpf[local_i].T

        start_i = end_i
        if progress_cb is not None:
            progress_cb(end_i, n_examples)

    return maps_by_example, img_by_example, path_by_example


@beartype.beartype
def get_precomputed_maps(
    artifacts: RunArtifacts,
    *,
    example_i: list[int],
    feature_i: list[int],
) -> tuple[dict[int, np.ndarray], dict[int, Image.Image], dict[int, str]]:
    msg = "Precomputed map mode requires sparse token acts."
    assert artifacts.token_acts_ts is not None, msg
    token_acts_ts = artifacts.token_acts_ts

    feat_pos = {feat_i: i for i, feat_i in enumerate(feature_i)}
    maps_by_example: dict[int, np.ndarray] = {
        ex_i: np.zeros((len(feature_i), artifacts.n_patches), dtype=np.float32)
        for ex_i in example_i
    }

    wanted = set(example_i)
    for feat_i in feature_i:
        start_i = int(token_acts_ts.indptr[feat_i])
        end_i = int(token_acts_ts.indptr[feat_i + 1])
        token_i = token_acts_ts.indices[start_i:end_i]
        vals = token_acts_ts.data[start_i:end_i]
        if vals.size == 0:
            continue

        ex_i_a = token_i // artifacts.n_patches
        patch_i_a = token_i % artifacts.n_patches
        slot_i = feat_pos[feat_i]
        for ex_i, patch_i, val in zip(
            ex_i_a.tolist(), patch_i_a.tolist(), vals.tolist(), strict=True
        ):
            if ex_i not in wanted:
                continue
            maps_by_example[ex_i][slot_i, patch_i] = float(val)

    img_by_example: dict[int, Image.Image] = {}
    path_by_example: dict[int, str] = {}
    for ex_i in example_i:
        path, _target = artifacts.dataset.samples[ex_i]
        path_by_example[ex_i] = path
        img = Image.open(path).convert("RGB")
        img_by_example[ex_i] = artifacts.resize(img)

    return maps_by_example, img_by_example, path_by_example


@beartype.beartype
def get_top_examples_from_sampled_maps(
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
        "Select a feature index range, pick top images per feature from sparse inference outputs, then recompute DINOv2 layer activations and SAE feature maps for patch overlays."
    )

    default_run = "/local/scratch/beattie.74/saev_demo/fish/saev/runs/5nprxikb"
    run_path = st.sidebar.text_input("Run directory", value=default_run)
    full_shard_path = st.sidebar.text_input(
        "Full inference shard path (optional)", value=""
    )
    shard_id = st.sidebar.text_input(
        "Shard id (used if full path is empty)", value="02b05f61"
    )
    dataset_source = st.sidebar.selectbox(
        "Dataset source",
        options=["From shard metadata", "Custom image folder"],
        index=0,
    )
    dataset_root = st.sidebar.text_input(
        "Dataset root (for custom image folder)",
        value="/local/scratch/beattie.74/fish_sae/fish-vista/shard_input",
    )
    feature_start = int(
        st.sidebar.number_input("Feature start index", min_value=0, value=0)
    )
    feature_end = int(
        st.sidebar.number_input("Feature end index (inclusive)", min_value=0, value=10)
    )
    n_top = int(
        st.sidebar.number_input(
            "Top images per feature", min_value=1, max_value=25, value=5
        )
    )
    batch_size = int(
        st.sidebar.number_input(
            "Recompute batch size", min_value=1, max_value=1024, value=64
        )
    )
    n_candidate_examples = int(
        st.sidebar.number_input(
            "Approx candidate examples",
            min_value=32,
            max_value=10000,
            value=1024,
            step=32,
        )
    )
    random_seed = int(
        st.sidebar.number_input("Approx random seed", min_value=0, value=0)
    )
    mode = st.sidebar.selectbox(
        "Activation source",
        options=[
            "Approx sampled recompute (fast load)",
            "Precomputed sparse (full, slower load)",
        ],
        index=0,
    )
    use_approx = mode == "Approx sampled recompute (fast load)"
    recompute = use_approx
    load_sparse_artifacts = not use_approx
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
    shard_id = shard_id.strip() or None
    inference_shard_dpath = None
    if full_shard_path.strip() != "":
        inference_shard_dpath = pathlib.Path(full_shard_path).expanduser().resolve()

    dataset_root_dpath = None
    if dataset_source == "Custom image folder":
        if dataset_root.strip() == "":
            st.error("Provide a dataset root path when using Custom image folder.")
            return
        dataset_root_dpath = pathlib.Path(dataset_root).expanduser().resolve()

    with st.spinner("Loading artifacts..."):
        artifacts = load_run_artifacts(
            run_dpath,
            shard_id,
            inference_shard_dpath,
            dataset_root_dpath,
            device,
            recompute=recompute,
            load_sparse_artifacts=load_sparse_artifacts,
        )

    topk_ex_i_sf: np.ndarray | None = None
    topk_scores_sf: np.ndarray | None = None
    if not use_approx:
        msg = "Full sparse mode requires token acts."
        assert artifacts.token_acts_ts is not None, msg
        with st.spinner("Loading or creating top-10 feature index cache..."):
            topk_ex_i_sf, topk_scores_sf = load_or_make_topk_examples(
                artifacts.inference_dpath,
                artifacts.token_acts_ts,
                n_patches=artifacts.n_patches,
                k=10,
            )

    if feature_end < feature_start:
        feature_start, feature_end = feature_end, feature_start
    feature_start = min(feature_start, artifacts.d_sae - 1)
    feature_end = min(feature_end, artifacts.d_sae - 1)
    features = list(range(feature_start, feature_end + 1))

    if not features:
        st.error("Feature range is empty after bounds checks.")
        return

    st.write(
        f"Visualizing features [{feature_start}, {feature_end}] ({len(features)} total)."
    )

    top_examples_by_feature: dict[int, list[tuple[int, float]]] = {}
    all_example_i: list[int] = []
    seen = set()

    if use_approx:
        n_dataset = len(artifacts.dataset)
        n_draw = min(n_candidate_examples, n_dataset)
        rng = np.random.default_rng(seed=random_seed)
        sampled_example_i = rng.choice(n_dataset, size=n_draw, replace=False).tolist()
        with st.spinner("Computing sampled DINOv2 + SAE activations..."):
            progress = st.progress(0.0, text=f"Recomputing activations: 0/{n_draw}")

            def on_progress(done: int, total: int) -> None:
                if total <= 0:
                    progress.progress(1.0, text="Recomputing activations: 0/0")
                    return
                frac = float(done) / float(total)
                progress.progress(
                    frac,
                    text=f"Recomputing activations: {done}/{total}",
                )

            maps_by_example, img_by_example, path_by_example = recompute_maps(
                artifacts,
                example_i=sampled_example_i,
                feature_i=features,
                device=device,
                batch_size=batch_size,
                progress_cb=on_progress,
            )

        feat_pos = {feat_i: i for i, feat_i in enumerate(features)}
        with st.spinner("Ranking top images within sampled activations..."):
            for feat_i in features:
                top_examples = get_top_examples_from_sampled_maps(
                    maps_by_example,
                    feature_i=feat_i,
                    feat_pos=feat_pos,
                    n_top=n_top,
                )
                top_examples_by_feature[feat_i] = top_examples
                for ex_i, _score in top_examples:
                    if ex_i in seen:
                        continue
                    seen.add(ex_i)
                    all_example_i.append(ex_i)

        st.caption(
            f"Approx mode: ranked features using {n_draw} sampled examples (seed={random_seed})."
        )
    else:
        msg = "Full sparse mode requires token acts."
        assert artifacts.token_acts_ts is not None, msg
        with st.spinner("Collecting top activating examples for selected features..."):
            for feat_i in features:
                top_examples = get_top_examples_for_feature(
                    artifacts.token_acts_ts,
                    feature_i=feat_i,
                    n_patches=artifacts.n_patches,
                    n_top=n_top,
                    topk_ex_i_sf=topk_ex_i_sf,
                    topk_scores_sf=topk_scores_sf,
                )
                top_examples_by_feature[feat_i] = top_examples
                for ex_i, _score in top_examples:
                    if ex_i in seen:
                        continue
                    seen.add(ex_i)
                    all_example_i.append(ex_i)

        with st.spinner("Building maps from precomputed sparse activations..."):
            maps_by_example, img_by_example, path_by_example = get_precomputed_maps(
                artifacts,
                example_i=all_example_i,
                feature_i=features,
            )

    feat_pos = {feat_i: i for i, feat_i in enumerate(features)}
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
                            f"recomputed_max={float(np.max(acts_p)):.4g}"
                        ),
                        use_container_width=True,
                    )
                    st.caption(path_by_example[ex_i])


if __name__ == "__main__":
    run_app()
