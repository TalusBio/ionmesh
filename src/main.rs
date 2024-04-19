// OrderedDict([
//     ('psm_id', Int64),
//     ('fragment_type', Utf8),
//     ('fragment_ordinals', Int32),
//     ('fragment_charge', Int32),
//     ('fragment_mz_calculated', Float32),
//     ('fragment_intensity', Float32)])
//

mod aggregation;
mod extraction;
mod mod_types;
mod ms;
mod scoring;
mod space;

mod utils;
mod visualization;

extern crate log;
extern crate pretty_env_logger;

use clap::Parser;

use crate::scoring::SageSearchConfig;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    config: String,
    #[arg(short, long, default_value = "peakachu_output")]
    output_dir: String,
    #[arg(long, action)]
    write_template: bool,
    /// File Path to use (.d)
    files: Vec<String>,
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

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
struct Config {
    denoise_config: DenoiseConfig,
    tracing_config: TracingConfig,
    pseudoscan_generation_config: PseudoscanGenerationConfig,
    sage_search_config: SageSearchConfig,
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

        let out_path = args.config;
        if fs::metadata(out_path.clone()).is_ok() {
            panic!("File already exists: {}", out_path);
        } else {
            std::fs::write(out_path.clone(), config_str).unwrap();
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

    let path_use = args.files;
    if path_use.len() != 1 {
        panic!("I have only implemented one path!!!");
    }
    let path_use = path_use[0].clone();
    // ms_denoise::read_all_ms1_denoising(path_use.clone(), &mut rec);

    let out_path_dir = Path::new(&args.output_dir);
    // Create dir if not exists ...
    if !out_path_dir.exists() {
        fs::create_dir_all(out_path_dir).unwrap();
    }
    let out_path_scans = out_path_dir.join("pseudoscans_debug.json");
    let out_path_features = out_path_dir.join("sage_features_debug.csv");
    let out_traces_path = out_path_dir.join("chr_traces_debug.csv");

    if true {
        log::info!("Reading DIA data from: {}", path_use);
        let (dia_frames, dia_info) = aggregation::ms_denoise::read_all_dia_denoising(
            path_use.clone(),
            config.denoise_config.ms2_min_n.into(),
            config.denoise_config.ms2_min_cluster_intensity.into(),
            config.denoise_config.mz_scaling.into(),
            config.denoise_config.ims_scaling,
            &mut rec,
        );

        let cycle_time = dia_info.calculate_cycle_time();

        let mut traces = aggregation::tracing::combine_traces(
            dia_frames,
            config.tracing_config.mz_scaling.into(),
            config.tracing_config.rt_scaling.into(),
            config.tracing_config.ims_scaling.into(),
            config.tracing_config.min_n.into(),
            config.tracing_config.min_neighbor_intensity,
            cycle_time as f32,
            &mut rec,
        );

        let out = aggregation::tracing::write_trace_csv(&traces, out_traces_path);
        match out {
            Ok(_) => {}
            Err(e) => {
                log::warn!("Error writing traces: {:?}", e);
            }
        }

        println!("traces: {:?}", traces.len());
        traces.retain(|x| x.num_agg > 5);
        traces.retain(|x| x.num_rt_points >= 2);
        println!("traces: {:?}", traces[traces.len()-5]);
        println!("traces: {:?}", traces.len());

        // Maybe reparametrize as 1.1 cycle time
        let pseudoscans = aggregation::tracing::combine_pseudospectra(
            traces,
            config.pseudoscan_generation_config.rt_scaling.into(),
            config.pseudoscan_generation_config.ims_scaling.into(),
            config.pseudoscan_generation_config.quad_scaling.into(),
            config.pseudoscan_generation_config.min_neighbor_intensity,
            config.pseudoscan_generation_config.min_n.into(),
            &mut rec,
        );

        // Report min/max/average/std and skew for ims and rt
        let ims_stats =
            utils::get_stats(&pseudoscans.iter().map(|x| x.ims as f64).collect::<Vec<_>>());
        let ims_sd_stats = utils::get_stats(
            &pseudoscans
                .iter()
                .map(|x| x.ims_std as f64)
                .collect::<Vec<_>>(),
        );
        let rt_stats =
            utils::get_stats(&pseudoscans.iter().map(|x| x.rt as f64).collect::<Vec<_>>());
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

        let out =
            aggregation::tracing::write_pseudoscans_json(&pseudoscans, out_path_scans.clone());
        match out {
            Ok(_) => {}
            Err(e) => {
                log::warn!("Error writing pseudoscans: {:?}", e);
            }
        }
    }

    let pseudoscans_read = aggregation::tracing::read_pseudoscans_json(out_path_scans);
    let pseudoscans = pseudoscans_read.unwrap();
    println!("pseudoscans: {:?}", pseudoscans.len());

    let score_out = scoring::score_pseudospectra(
        pseudoscans,
        config.sage_search_config,
        out_path_features.clone(),
    );
    match score_out {
        Ok(_) => {}
        Err(e) => {
            log::error!("Error scoring pseudospectra: {:?}", e);
        }
    }
}
