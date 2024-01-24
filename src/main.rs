// OrderedDict([
//     ('psm_id', Int64),
//     ('fragment_type', Utf8),
//     ('fragment_ordinals', Int32),
//     ('fragment_charge', Int32),
//     ('fragment_mz_calculated', Float32),
//     ('fragment_intensity', Float32)])
//

mod dbscan;
mod extraction;
mod kdtree;
mod mod_types;
mod ms;
mod ms_denoise;
mod quad;
mod scoring;
mod space_generics;
mod tdf;
mod trace_combination;
mod tracing;
mod utils;
mod visualization;

extern crate log;
extern crate pretty_env_logger;

use clap::Parser;
use log::info;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File Path to use (.d)
    #[arg(short, long)]
    file: String,
    #[arg(short, long)]
    config: String,
    #[arg(long, action)]
    write_template: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
struct TracingConfig {
    mz_scaling: f32,
    rt_scaling: f32,
    ims_scaling: f32,
    min_n: u8,
    min_neighbor_intensity: u32,
}

impl Default for TracingConfig {
    fn default() -> Self {
        TracingConfig {
            mz_scaling: 0.02,
            rt_scaling: 2.2,
            ims_scaling: 0.02,
            min_n: 2,
            min_neighbor_intensity: 200,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
struct DenoiseConfig {
    mz_scaling: f32,
    ims_scaling: f32,
    ms2_min_n: u8,
    ms1_min_n: u8,
    ms1_min_cluster_intensity: u32,
    ms2_min_cluster_intensity: u32,
}

impl Default for DenoiseConfig {
    fn default() -> Self {
        DenoiseConfig {
            mz_scaling: 0.02,
            ims_scaling: 0.02,
            ms2_min_n: 2,
            ms1_min_n: 3,
            ms1_min_cluster_intensity: 100,
            ms2_min_cluster_intensity: 100,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
struct PseudoscanGenerationConfig {
    rt_scaling: f32,
    quad_scaling: f32,
    ims_scaling: f32,
    min_n: u8,
    min_neighbor_intensity: u32,
}

impl Default for PseudoscanGenerationConfig {
    fn default() -> Self {
        PseudoscanGenerationConfig {
            rt_scaling: 2.2,
            quad_scaling: 5.,
            ims_scaling: 0.02,
            min_n: 5,
            min_neighbor_intensity: 200,
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, Copy)]
struct Config {
    denoise_config: DenoiseConfig,
    tracing_config: TracingConfig,
    pseudoscan_generation_config: PseudoscanGenerationConfig,
}

impl Config {
    fn from_toml(path: String) -> Result<Self, Box<dyn std::error::Error>> {
        let config_str = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&config_str)?;
        Ok(config)
    }
}

fn main() {
    let args = Args::parse();

    if args.write_template {
        let config = Config::default();
        let config_str = toml::to_string_pretty(&config).unwrap();

        let out_path = "default_peakachu_config.toml";
        if fs::metadata(out_path).is_ok() {
            panic!("File already exists: {}", out_path);
        } else {
            std::fs::write(out_path, config_str).unwrap();
            println!("Wrote default config to {}", out_path);
            return;
        }
    }

    // Parse the config from toml
    let config = Config::from_toml(args.config).unwrap();

    pretty_env_logger::init();

    let mut rec: Option<rerun::RecordingStream> = None;
    if cfg!(feature = "viz") {
        rec = Some(visualization::setup_recorder());
    }

    let path_use = args.file;
    // ms_denoise::read_all_ms1_denoising(path_use.clone(), &mut rec);

    if true {
        let dia_frames = ms_denoise::read_all_dia_denoising(
            path_use.clone(),
            config.denoise_config.ms2_min_n.into(),
            config.denoise_config.ms2_min_cluster_intensity.into(),
            config.denoise_config.mz_scaling.into(),
            config.denoise_config.ims_scaling.into(),
            &mut rec,
        );

        let traces = tracing::combine_traces(
            dia_frames,
            config.tracing_config.mz_scaling.into(),
            config.tracing_config.rt_scaling.into(),
            config.tracing_config.ims_scaling.into(),
            config.tracing_config.min_n.into(),
            config.tracing_config.min_neighbor_intensity.into(),
            &mut rec,
        );

        let out = tracing::write_trace_csv(&traces, &"traces_debug.csv".into());
        match out {
            Ok(_) => {}
            Err(e) => {
                log::warn!("Error writing traces: {:?}", e);
            }
        }

        // Maybe reparametrize as 1.1 cycle time
        let pseudoscans = tracing::combine_pseudospectra(
            traces,
            config.pseudoscan_generation_config.rt_scaling.into(),
            config.pseudoscan_generation_config.ims_scaling.into(),
            config.pseudoscan_generation_config.quad_scaling.into(),
            config
                .pseudoscan_generation_config
                .min_neighbor_intensity
                .into(),
            config.pseudoscan_generation_config.min_n.into(),
            &mut rec,
        );

        // Report min/max/average/std and skew for ims and rt
        let ims_stats = utils::get_stats(&pseudoscans.iter().map(|x| x.ims as f64).collect::<Vec<_>>());
        let ims_sd_stats = utils::get_stats(
            &pseudoscans
                .iter()
                .map(|x| x.ims_std as f64)
                .collect::<Vec<_>>(),
        );
        let rt_stats = utils::get_stats(&pseudoscans.iter().map(|x| x.rt as f64).collect::<Vec<_>>());
        let rt_sd_stats = utils::get_stats(
            &pseudoscans
                .iter()
                .map(|x| x.rt_std as f64)
                .collect::<Vec<_>>(),
        );
        let npeaks = utils::get_stats(
            &pseudoscans
                .iter()
                .map(|x| x.peaks.len() as f64)
                .collect::<Vec<_>>(),
        );

        println!("ims_stats: {:?}", ims_stats);
        println!("rt_stats: {:?}", rt_stats);

        println!("ims_sd_stats: {:?}", ims_sd_stats);
        println!("rt_sd_stats: {:?}", rt_sd_stats);

        println!("npeaks: {:?}", npeaks);

        let out = tracing::write_pseudoscans_json(&pseudoscans, r"pseudoscans_debug.json".into());
        match out {
            Ok(_) => {}
            Err(e) => {
                log::warn!("Error writing pseudoscans: {:?}", e);
            }
        }
    }

    let pseudoscans_read = tracing::read_pseudoscans_json(r"pseudoscans_debug.json".into());
    let pseudoscans = pseudoscans_read.unwrap();
    println!("pseudoscans: {:?}", pseudoscans.len());

    let fasta_path = "/Users/sebastianpaez/git/2023_dev_diadem_report/data/UP000005640_9606.fasta";
    let score_out = scoring::score_pseudospectra(pseudoscans, fasta_path.into());
    match score_out {
        Ok(_) => {}
        Err(e) => {
            log::error!("Error scoring pseudospectra: {:?}", e);
        }
    }
}
