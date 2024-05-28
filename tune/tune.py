import optuna
import tomli_w
import tomllib
import subprocess
from loguru import logger
import argparse
from pathlib import Path
from dataclasses import dataclass
import random
import polars as pl
import json
from matplotlib import pyplot as plt
import numpy as np
import time
import pprint

# DEBUG_TRACES_FROM_CACHE


@dataclass
class IonMeshTuner:
    fasta: Path
    base_config: Path
    target_file: Path
    ionmesh_executable: Path
    base_out_dir: Path = Path("tune_out")

    def objective(self, trial: optuna.Trial):
        id = trial.number
        base_toml = tomllib.load(open(self.base_config, "rb"))
        tempdir = self.base_out_dir / f"{id}_{random.randint(0, 1000000)}"
        while tempdir.exists():
            tempdir = self.base_out_dir / f"{id}_{random.randint(0, 1000000)}"

        logger.info(f"Running with {tempdir}")

        tempdir.mkdir(exist_ok=True, parents=True)

        denoise_mz_scale = trial.suggest_float("dn_mz", 0.005, 0.03)
        denoise_ims_scale = trial.suggest_float("dn_ims", 0.005, 0.03)
        base_toml["denoise_config"]["mz_scaling"] = denoise_mz_scale
        base_toml["denoise_config"]["ims_scaling"] = denoise_ims_scale

        tracing_mz_scale = trial.suggest_float("t_mz", 0.01, 0.02)
        tracing_ims_scale = trial.suggest_float("t_ims", 0.01, 0.03)
        tracing_rt_scale = trial.suggest_float("t_rt", 0.5, 2.5)
        tracing_min_intensity = trial.suggest_int("t_min", 50, 5000)
        base_toml["tracing_config"]["mz_scaling"] = tracing_mz_scale
        base_toml["tracing_config"]["ims_scaling"] = tracing_ims_scale
        base_toml["tracing_config"]["rt_scaling"] = tracing_rt_scale
        base_toml["tracing_config"]["min_neighbor_intensity"] = tracing_min_intensity

        pseudoscan_rt_scale = trial.suggest_float("ps_rt", 0.5, 2.5)
        pseudoscan_ims_scale = trial.suggest_float("ps_ims", 0.01, 0.03)
        pseudoscan_min_intensity = trial.suggest_int("ps_min", 50, 10000)
        base_toml["pseudoscan_generation_config"]["rt_scaling"] = pseudoscan_rt_scale
        base_toml["pseudoscan_generation_config"]["ims_scaling"] = pseudoscan_ims_scale
        base_toml["pseudoscan_generation_config"]["min_neighbor_intensity"] = (
            pseudoscan_min_intensity
        )

        base_toml["sage_search_config"]["fasta_path"] = str(self.fasta)

        temp_toml = tempdir / "temp.toml"
        tomli_w.dump(base_toml, open(temp_toml, "wb"))

        logger.info(f"Running with config: {pprint.pformat(base_toml)}")

        run_start = time.time()
        try:
            subprocess.run(
                [
                    str(self.ionmesh_executable),
                    "--config",
                    str(temp_toml),
                    "--output-dir",
                    tempdir,
                    str(self.target_file),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running IonMesh: {e}")
            return float("nan")

        run_end = time.time()
        logger.info(f"Run time: {run_end - run_start}")

        delta_discriminant = self.read_results_dir(tempdir)
        return delta_discriminant

    def read_results_dir(self, results_dir):
        logger.info(f"Reading results from {results_dir}")
        pseudoscan_df = pl.DataFrame(
            json.load(open(str(results_dir / "debug_scans.json"))), strict=False
        )
        (results_dir / "debug_scans.json").unlink()

        logger.info("Converting scans to parquet")
        pseudoscan_df.write_parquet(str(results_dir / "debug_scans.parquet"))
        del pseudoscan_df

        # Still pretty big ... I will delete it for now
        # pl.scan_csv(str(results_dir / "debug_traces.csv")).sink_parquet(
        #     str(results_dir / "debug_traces.parquet")
        # )
        (results_dir / "debug_traces.csv").unlink()

        logger.info("Reading features/scores")
        sage_features = pl.scan_csv(results_dir / "features.csv")
        sage_features.sink_parquet(results_dir / "features.parquet")
        (results_dir / "features.csv").unlink()
        sage_features = pl.scan_parquet(results_dir / "features.parquet")

        # ['peptide', 'psm_id', 'peptide_len', 'spec_id', 'file_id',
        # 'rank', 'label', 'expmass', 'calcmass', 'charge', 'rt', 'aligned_rt', 'predicted_rt',
        # 'delta_rt_model', 'delta_mass', 'isotope_error', 'average_ppm', 'hyperscore',
        # 'delta_next', 'delta_best', 'matched_peaks', 'longest_b', 'longest_y', 'longest_y_pct',
        # 'missed_cleavages', 'matched_intensity_pct', 'scored_candidates', 'poisson',
        # 'discriminant_score', 'posterior_error', 'spectrum_q', 'peptide_q', 'protein_q',
        # 'ms2_intensity']

        # Keep best scoring target for each peptide
        TARTGET_SCORE = "hyperscore"
        sage_features = sage_features.sort(TARTGET_SCORE, descending=True)
        sage_features = sage_features.group_by("peptide").head(1)

        targets = sage_features.filter(pl.col("rank") == 1, pl.col("label") == 1)
        decoys = sage_features.filter(pl.col("rank") == 1, pl.col("label") < 0)

        target_discriminant = targets.select(TARTGET_SCORE).collect()
        target_discriminant_sum = target_discriminant[TARTGET_SCORE].sum()
        decoy_discriminant = decoys.select(TARTGET_SCORE).collect()
        decoy_discriminant_sum = decoy_discriminant[TARTGET_SCORE].sum()

        fig, ax = plt.subplots()
        bins = np.histogram_bin_edges(
            np.concatenate(
                [
                    target_discriminant[TARTGET_SCORE],
                    decoy_discriminant[TARTGET_SCORE],
                ]
            ),
            bins=100,
        )
        ax.hist(
            target_discriminant[TARTGET_SCORE],
            bins=bins,
            alpha=0.5,
            label="Target",
        )
        ax.hist(
            decoy_discriminant[TARTGET_SCORE],
            bins=bins,
            alpha=0.5,
            label="Decoy",
        )
        ax.legend()
        plt.savefig(results_dir / f"{TARTGET_SCORE}_histogram.png")

        logger.info(
            f"Target {TARTGET_SCORE}: {target_discriminant_sum} / {len(target_discriminant)}"
        )
        logger.info(
            f"Decoy {TARTGET_SCORE}: {decoy_discriminant_sum} / {len(decoy_discriminant)}"
        )

        delta_discriminant = target_discriminant_sum - decoy_discriminant_sum

        return delta_discriminant


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path)
    parser.add_argument("--base_config", type=Path)
    parser.add_argument("--target_file", type=Path)
    parser.add_argument("--ionmesh_executable", type=Path)
    parser.add_argument("--output", type=Path, default=Path("tune_out"))
    return parser


def main():
    parser = build_parser()
    args, unkargs = parser.parse_known_args()
    if unkargs:
        raise ValueError(f"Unknown arguments: {unkargs}")

    tuner = IonMeshTuner(
        fasta=args.fasta,
        base_config=args.base_config,
        target_file=args.target_file,
        ionmesh_executable=args.ionmesh_executable,
        base_out_dir=args.output,
    )

    study = optuna.create_study(
        storage="sqlite:///tune.db",
        study_name="ionmesh",
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(tuner.objective, n_trials=100)
    logger.info(study.best_params)


if __name__ == "__main__":
    main()
