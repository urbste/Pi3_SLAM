import argparse
import csv
import json
import os
import time

import numpy as np
import pytheia as pt
import matplotlib.pyplot as plt


def reprojection_error(recon: pt.sfm.Reconstruction) -> float:
    total_error = 0.0
    total_obs = 0
    for view_id in recon.ViewIds():
        view = recon.View(view_id)
        for track_id in view.TrackIds():
            track = recon.Track(track_id)
            if not track.IsEstimated():
                continue
            # Project homogeneous 3D point into the camera; API returns (success, uv)
            _, uv = view.Camera().ProjectPoint(track.Point())

            feat = view.GetFeature(track_id)
            err = np.linalg.norm(feat.point - uv)
            total_error += float(err)
            total_obs += 1
    return (total_error / max(1, total_obs))


def build_tests() -> dict:
    S = pt.sfm.LinearSolverType
    P = pt.sfm.PreconditionerType
    V = pt.sfm.VisibilityClusteringType
    return {
        # Baselines
        # "dense_schur_identity_single": dict(
        #     linear_solver=S.DENSE_SCHUR, precond=P.IDENTITY, vis=V.CANONICAL_VIEWS,
        #     homo=False, inv_depth=False, max_iters=50
        # ),

        # "iterative_schur_jacobi_canonical": dict(
        #     linear_solver=S.ITERATIVE_SCHUR, precond=P.JACOBI, vis=V.CANONICAL_VIEWS,
        #     homo=False, inv_depth=False, max_iters=50
        # ),

        # Mixed-precision tests (max_num_iterations=2)
        # Dense solvers: DENSE_NORMAL_CHOLESKY and DENSE_SCHUR
        # "dense_schur_mp2": dict(
        #     linear_solver=S.DENSE_SCHUR, precond=P.IDENTITY, vis=V.SINGLE_LINKAGE,
        #     homo=False, inv_depth=False, max_iters=50, mixed_precision=True, refine_iters=1
        # ), 
        # "dense_schur_mp2": dict(
        #     linear_solver=S.DENSE_SCHUR, precond=P.IDENTITY, vis=V.CANONICAL_VIEWS,
        #     homo=True, inv_depth=False, max_iters=50, mixed_precision=True, refine_iters=1
        # ),
        "dense_schur_mp2": dict(
            linear_solver=S.DENSE_SCHUR, precond=P.IDENTITY, vis=V.CANONICAL_VIEWS,
            homo=False, inv_depth=True, max_iters=50, mixed_precision=True, refine_iters=1
        ),
        # Parameterizations
        "sparse_schur_homogeneous": dict(
            linear_solver=S.SPARSE_SCHUR, precond=P.SCHUR_JACOBI, vis=V.CANONICAL_VIEWS,
            homo=True, inv_depth=False, max_iters=50
        ),
        "dense_schur_identity_canonical": dict(
            linear_solver=S.DENSE_SCHUR, precond=P.IDENTITY, vis=V.CANONICAL_VIEWS,
            homo=True, inv_depth=False, max_iters=50
        ),
        "sparse_schur_inverse_depth": dict(
            linear_solver=S.SPARSE_SCHUR, precond=P.SCHUR_JACOBI, vis=V.CANONICAL_VIEWS,
            homo=False, inv_depth=True, max_iters=50
        ),
        "dense_schur_cluster_tridiag_canonical_inv_depth": dict(
            linear_solver=S.DENSE_SCHUR, precond=P.IDENTITY, vis=V.CANONICAL_VIEWS,
            homo=False, inv_depth=True, max_iters=50
        ),
    }


def run_test_once(recon_path: str, cfg: dict) -> dict:
    recon = pt.io.ReadReconstruction(recon_path)
    try:
        # Some bindings may return (status, reconstruction)
        if isinstance(recon, (list, tuple)):
            recon = recon[-1]
    except Exception:
        pass

    ba_opts = pt.sfm.BundleAdjustmentOptions()
    ba_opts.use_orientation_priors = True
    ba_opts.use_position_priors = True
    ba_opts.max_num_iterations = int(cfg.get("max_iters", 50))
    ba_opts.verbose = False
    ba_opts.linear_solver_type = cfg["linear_solver"]
    ba_opts.preconditioner_type = cfg["precond"]
    ba_opts.visibility_clustering_type = cfg["vis"]
    ba_opts.use_homogeneous_point_parametrization = bool(cfg.get("homo", False))
    ba_opts.use_inverse_depth_parametrization = bool(cfg.get("inv_depth", False))
    ba_opts.use_mixed_precision_solves = bool(cfg.get("mixed_precision", False))
    ba_opts.max_num_refinement_iterations = int(cfg.get("refine_iters", 0))
    ba_opts.use_inner_iterations = False


    err_before = reprojection_error(recon)

    start_time = time.perf_counter()

    recon.InitializeInverseDepth()
    print(f"   Initialized inverse depth in C++ {time.perf_counter() - start_time:.3f}s")


    t0 = time.perf_counter()
    summary = pt.sfm.BundleAdjustReconstruction(ba_opts, recon)
    total_s = time.perf_counter() - t0

    err_after = reprojection_error(recon)


    result = dict(
        success=bool(getattr(summary, "success", False)),
        initial_cost=float(getattr(summary, "initial_cost", 0.0)),
        final_cost=float(getattr(summary, "final_cost", 0.0)),
        setup_time_s=float(getattr(summary, "setup_time_in_seconds", 0.0)),
        solve_time_s=float(getattr(summary, "solve_time_in_seconds", 0.0)),
        total_time_s=float(total_s),
        reprojection_error_before=float(err_before),
        reprojection_error_after=float(err_after),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate BA configurations and log reprojection error, timings, and iterations")
    parser.add_argument("--reconstruction", default="recon_qry_before_ba.sfm", help="Path to input .sfm reconstruction")
    parser.add_argument("--output", default="ba_results", help="Directory to write results (CSV/JSON)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    tests = build_tests()

    rows = []
    json_out = {}
    for name, cfg in tests.items():
        print(f"\n▶️  Running test: {name}")
        res = run_test_once(args.reconstruction, cfg)
        print(f"   success={res['success']}  final_cost={res['final_cost']:.6f}  err_before={res['reprojection_error_before']:.4f}  err_after={res['reprojection_error_after']:.4f}  time={res['total_time_s']:.3f}s")

        row = dict(
            test=name,
            linear_solver=str(cfg["linear_solver"]).split(".")[-1],
            preconditioner=str(cfg["precond"]).split(".")[-1],
            visibility=str(cfg["vis"]).split(".")[-1],
            homo=bool(cfg.get("homo", False)),
            inv_depth=bool(cfg.get("inv_depth", False)),
            max_iters=int(cfg.get("max_iters", 50)),
            mixed_precision=bool(cfg.get("mixed_precision", False)),
            refine_iters=int(cfg.get("refine_iters", 0)),
            **res,
        )
        rows.append(row)
        json_out[name] = row

    # Write CSV
    csv_path = os.path.join(args.output, "ba_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n💾 Wrote CSV: {csv_path}")

    # Write JSON
    json_path = os.path.join(args.output, "ba_results.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"💾 Wrote JSON: {json_path}")

    # Plot summary
    try:
        names = [r['test'] for r in rows]
        err_bef = np.array([r['reprojection_error_before'] for r in rows], dtype=float)
        err_aft = np.array([r['reprojection_error_after'] for r in rows], dtype=float)
        times = np.array([r['total_time_s'] for r in rows], dtype=float)
        final_costs = np.array([r['final_cost'] for r in rows], dtype=float)
        iterations = np.array([r['iterations'] if r['iterations'] is not None else np.nan for r in rows], dtype=float)

        x = np.arange(len(names))
        width = 0.35

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)

        # (1) Reprojection error before/after
        ax = axes[0]
        ax.bar(x - width/2, err_bef, width, label='reproj before')
        ax.bar(x + width/2, err_aft, width, label='reproj after')
        ax.set_title('Reprojection error (avg)')
        ax.set_ylabel('pixels')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
        ax.legend()

        # (2) Final cost
        ax = axes[1]
        ax.bar(x, final_costs, color='#6aa84f')
        ax.set_title('Final cost')
        ax.set_ylabel('cost')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)

        # (3) Total time and iterations
        ax = axes[2]
        ax2 = ax.twinx()
        ax.bar(x, times, color='#3c78d8', alpha=0.8, label='total time (s)')
        ax.set_title('Runtime')
        ax.set_ylabel('seconds')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
        # Compose legend
        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc='upper right')

        plot_path = os.path.join(args.output, 'ba_results.png')
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"🖼️  Wrote plot: {plot_path}")
    except Exception as e:
        print(f"⚠️  Failed to create plot: {e}")


if __name__ == "__main__":
    main()








