// OrderedDict([
//     ('psm_id', Int64),
//     ('fragment_type', Utf8),
//     ('fragment_ordinals', Int32),
//     ('fragment_charge', Int32),
//     ('fragment_mz_calculated', Float32),
//     ('fragment_intensity', Float32)])
//

mod aggregation;
mod ms;
mod scoring;
mod space;

mod utils;

extern crate log;
extern crate pretty_env_logger;

use std::fs;
use std::io::Error;
use std::path::Path;

use clap::Parser;
use log::debug;
use serde::{
    Deserialize,
    Serialize,
};

use crate::scoring::SageSearchConfig;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    config: String,
    #[arg(short, long, default_value = "ionmesh_output")]
    output_dir: String,
    #[arg(long, action)]
    write_template: bool,
    /// File Path to use (.d)
    files: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct OutputConfig {
    //
    debug_scans_json: Option<String>,
    debug_traces_csv: Option<String>,
    out_features_csv: Option<String>,
}

impl Default for OutputConfig {
    fn default() -> Self {
        OutputConfig {
            debug_scans_json: None,
            debug_traces_csv: None,
            out_features_csv: Some("".into()),
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
struct Config {
    denoise_config: aggregation::ms_denoise::DenoiseConfig,
    tracing_config: aggregation::tracing::TracingConfig,
    pseudoscan_generation_config: aggregation::pseudospectra::PseudoscanGenerationConfig,
    sage_search_config: SageSearchConfig,
    output_config: OutputConfig,
}

impl Config {
    fn from_toml(path: String) -> Result<Self, Box<dyn std::error::Error>> {
        let config_str = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&config_str)?;
        Ok(config)
    }
}

fn main() -> Result<(), std::io::Error> {
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
            return Ok(());
        }
    }

    // Parse the config from toml
    let config = Config::from_toml(args.config).unwrap();

    pretty_env_logger::init();

    let path_use = args.files;
    if path_use.len() != 1 {
        panic!("I have only implemented one path!!!");
    }
    let path_use = path_use[0].clone();
    // ms_denoise::read_all_ms1_denoising(path_use.clone());

    let out_path_dir = Path::new(&args.output_dir);
    // Create dir if not exists ...
    if !out_path_dir.exists() {
        fs::create_dir_all(out_path_dir).unwrap();
    }

    // TODO: consier moving this to the config struct as an implementation.
    let out_path_scans = config
        .output_config
        .debug_scans_json
        .as_ref()
        .map(|path| out_path_dir.join(path).to_path_buf());
    let _out_traces_path = config
        .output_config
        .debug_traces_csv
        .as_ref()
        .map(|path| out_path_dir.join(path).to_path_buf());
    let out_path_features = config
        .output_config
        .out_features_csv
        .as_ref()
        .map(|path| out_path_dir.join(path).to_path_buf());

    log::info!("Reading DIA data from: {}", path_use);
    let tmp =
        aggregation::ms_denoise::read_all_dia_denoising(path_use.clone(), config.denoise_config);

    let (dia_frames, dia_info) = match tmp {
        Ok(x) => x,
        Err(e) => {
            log::error!("Error reading DIA data: {:?}", e);
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ));
        },
    };

    // TODO add here expansion limits
    let mut traces = aggregation::tracing::combine_traces(dia_frames, config.tracing_config);

    // let out = match out_traces_path {
    //     Some(out_path) => aggregation::tracing::write_trace_csv(&traces, out_path),
    //     None => Ok(()),
    // };
    // match out {
    //     Ok(_) => {},
    //     Err(e) => {
    //         log::warn!("Error writing traces: {:?}", e);
    //     },
    // }

    let num_traces = traces.len();
    for (i, trace) in traces.iter_mut().enumerate() {
        debug!("trace {}/{}: {}", i, num_traces, trace.len());
        trace.retain(|x| x.num_agg > 3);
        debug!(
            "trace {}/{}: {} (after dopping too short)",
            i,
            num_traces,
            trace.len()
        );
        if trace.len() > 5 {
            debug!("sample_trace: {:?}", trace[trace.len() - 4])
        }
    }

    // Maybe reparametrize as 1.1 cycle time
    // TODO add here expansion limits
    let mut pseudoscans = aggregation::pseudospectra::combine_pseudospectra(
        traces,
        config.pseudoscan_generation_config,
    );

    // Report min/max/average/std and skew for ims and rt
    // This can probably be a macro ...
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

    let out = match out_path_scans {
        Some(out_path) => {
            aggregation::pseudospectra::write_pseudoscans_json(&pseudoscans, out_path)
        },
        None => Ok(()),
    };

    match out {
        Ok(_) => {},
        Err(e) => {
            log::warn!("Error writing pseudoscans: {:?}", e);
        },
    }

    println!("pseudoscans: {:?}", pseudoscans.len());
    pseudoscans.retain(|x| x.peaks.len() > 5);

    let score_out = scoring::score_pseudospectra(
        pseudoscans,
        config.sage_search_config,
        out_path_features.clone(),
        1,
    )?;

    Ok(())
}
