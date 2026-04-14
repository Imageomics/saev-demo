"""Pareto analysis for SAE runs: k (L0 proxy) vs training MSE.

This script scans a runs root directory, reads each run's BatchTopK value (k) from
checkpoint config, fetches training/eval MSE from WandB summary, computes Pareto
optimal runs, and saves a scatter plot plus CSV.
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import wandb


def get_k_from_cfg(cfg: dict) -> int | None:
    act_cfg = cfg.get("sae", {}).get("activation", {})
    if not isinstance(act_cfg, dict):
        return None
    if act_cfg.get("key") != "batch-top-k":
        return None
    top_k = act_cfg.get("top_k")
    if not isinstance(top_k, int):
        return None
    return top_k


def get_mse_from_summary(summary: dict, preferred_key: str) -> tuple[float | None, str]:
    candidates = [
        preferred_key,
        "eval/mse",
        "loss/mse",
        "train/mse",
        "mse",
    ]
    for key in candidates:
        val = summary.get(key)
        if isinstance(val, int | float):
            return float(val), key
    return None, ""


def is_pareto_front(k_vals: np.ndarray, mse_vals: np.ndarray) -> np.ndarray:
    order = np.lexsort((mse_vals, k_vals))
    out = np.zeros(k_vals.shape[0], dtype=bool)
    best_mse = np.inf
    for idx in order.tolist():
        mse = mse_vals[idx]
        if mse <= best_mse:
            out[idx] = True
            best_mse = mse
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=pathlib.Path,
        default=pathlib.Path("/local/scratch/beattie.74/saev_demo/saev/runs"),
        help="Directory containing SAE run folders.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="",
        help="WandB entity. If empty, use api.default_entity.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="",
        help="Override WandB project. If empty, use each run config's wandb_project.",
    )
    parser.add_argument(
        "--mse-key",
        type=str,
        default="eval/mse",
        help="Preferred WandB summary key for MSE.",
    )
    parser.add_argument(
        "--out-png",
        type=pathlib.Path,
        default=pathlib.Path("demo/pareto_k_vs_mse.png"),
        help="Output plot path.",
    )
    parser.add_argument(
        "--out-csv",
        type=pathlib.Path,
        default=pathlib.Path("demo/pareto_k_vs_mse.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate plot points with run IDs.",
    )
    args = parser.parse_args()

    runs_root = args.runs_root.expanduser().resolve()
    msg = f"Runs root does not exist: '{runs_root}'."
    assert runs_root.is_dir(), msg

    api = wandb.Api()
    entity = args.entity or api.default_entity
    msg = "No WandB entity resolved. Set --entity or configure WandB login/default entity."
    assert isinstance(entity, str) and entity, msg

    rows: list[dict] = []
    run_dirs = sorted([d for d in runs_root.iterdir() if d.is_dir()])

    for run_dpath in run_dirs:
        cfg_fpath = run_dpath / "checkpoint" / "config.json"
        if not cfg_fpath.exists():
            continue

        try:
            with open(cfg_fpath) as fd:
                cfg = json.load(fd)
        except (OSError, json.JSONDecodeError):
            continue

        k = get_k_from_cfg(cfg)
        if k is None:
            continue

        run_id = run_dpath.name
        project = args.project or cfg.get("wandb_project", "saev")

        try:
            run = api.run(f"{entity}/{project}/{run_id}")
            summary = dict(run.summary)
        except Exception:
            continue

        mse, mse_key = get_mse_from_summary(summary, args.mse_key)
        if mse is None:
            continue

        rows.append({
            "run_id": run_id,
            "k": k,
            "mse": mse,
            "mse_key": mse_key,
            "entity": entity,
            "project": project,
        })

    msg = "No runs with both BatchTopK (k) and MSE found."
    assert rows, msg

    k_vals = np.array([r["k"] for r in rows], dtype=np.float64)
    mse_vals = np.array([r["mse"] for r in rows], dtype=np.float64)
    pareto_mask = is_pareto_front(k_vals, mse_vals)

    for i, is_pareto in enumerate(pareto_mask.tolist()):
        rows[i]["is_pareto"] = is_pareto

    rows.sort(key=lambda r: (r["k"], r["mse"], r["run_id"]))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w") as fd:
        fd.write("run_id,k,mse,mse_key,is_pareto,entity,project\n")
        for r in rows:
            fd.write(
                f"{r['run_id']},{r['k']},{r['mse']:.10g},{r['mse_key']},{r['is_pareto']},{r['entity']},{r['project']}\n"
            )

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.scatter(k_vals, mse_vals, alpha=0.75, label="runs")

    pareto_rows = [r for r in rows if r["is_pareto"]]
    pareto_rows.sort(key=lambda r: (r["k"], r["mse"]))
    pareto_k = np.array([r["k"] for r in pareto_rows], dtype=np.float64)
    pareto_mse = np.array([r["mse"] for r in pareto_rows], dtype=np.float64)

    ax.scatter(
        pareto_k,
        pareto_mse,
        color="tab:red",
        s=55,
        zorder=3,
        label="pareto",
    )
    ax.plot(pareto_k, pareto_mse, color="tab:red", linewidth=1.4, alpha=0.9)

    if args.annotate:
        for r in rows:
            ax.annotate(r["run_id"], (r["k"], r["mse"]), fontsize=7, alpha=0.8)

    ax.set_title("SAE Pareto Analysis: k (L0 proxy) vs MSE")
    ax.set_xlabel("k (BatchTopK, L0 proxy)")
    ax.set_ylabel("Average MSE")
    ax.grid(alpha=0.25)
    ax.legend()

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=180)
    plt.close(fig)

    n_pareto = int(sum(r["is_pareto"] for r in rows))
    print(f"Loaded {len(rows)} runs; Pareto-optimal: {n_pareto}.")
    print(f"CSV: {args.out_csv}")
    print(f"Plot: {args.out_png}")


if __name__ == "__main__":
    main()
