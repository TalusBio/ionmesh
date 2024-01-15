// OrderedDict([
//     ('psm_id', Int64),
//     ('fragment_type', Utf8),
//     ('fragment_ordinals', Int32),
//     ('fragment_charge', Int32),
//     ('fragment_mz_calculated', Float32),
//     ('fragment_intensity', Float32)])
//

use apache_avro::{Codec, Error, Schema, Writer};
mod dbscan;
mod extraction;
mod kdtree;
mod mod_types;
mod ms;
mod ms_denoise;
mod quad;
mod space_generics;
mod tdf;
mod tracing;
mod utils;
mod visualization;

extern crate pretty_env_logger;
#[macro_use]
extern crate log;

use clap::Parser;


#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File Path to use (.d)
    #[arg(short, long)]
    file: String,
}


fn main() -> Result<(), Error> {
    let args = Args::parse();

    pretty_env_logger::init();

    let mut rec: Option<rerun::RecordingStream> = None;
    if cfg!(feature = "viz") {
        rec = Some(visualization::setup_recorder());
    }

    let path_use = args.file;
    // ms_denoise::read_all_ms1_denoising(path_use.clone(), &mut rec);
    let dia_frames = ms_denoise::read_all_dia_denoising(path_use.clone(), &mut rec);

    let mz_scaling = 0.015;
    let rt_scaling = 2.;
    let ims_scaling = 0.015;

    let traces = tracing::combine_traces(dia_frames, mz_scaling, rt_scaling, ims_scaling, &mut rec);

    let quad_scaling = 5.;
    let pseudoscans =
        tracing::combine_pseudospectra(traces, rt_scaling, ims_scaling, quad_scaling, &mut rec);

    // Report min/max/average/std and skew for ims and rt
    let ims_stats = utils::get_stats(&pseudoscans.iter().map(|x| x.ims).collect::<Vec<_>>());
    let ims_sd_stats = utils::get_stats(&pseudoscans.iter().map(|x| x.ims_std).collect::<Vec<_>>());
    let rt_stats = utils::get_stats(&pseudoscans.iter().map(|x| x.rt).collect::<Vec<_>>());
    let rt_sd_stats = utils::get_stats(&pseudoscans.iter().map(|x| x.rt_std).collect::<Vec<_>>());
    let npeaks = utils::get_stats(
        &pseudoscans
            .iter()
            .map(|x| x.peaks.len() as f64)
            .collect::<Vec<_>>(),
    );

    // 1. Calculate IOU of the ranges between mean +/- 2(std)
    // 2. Cluster the peaks using graph clustering.

    println!("ims_stats: {:?}", ims_stats);
    println!("rt_stats: {:?}", rt_stats);

    println!("ims_sd_stats: {:?}", ims_sd_stats);
    println!("rt_sd_stats: {:?}", rt_sd_stats);

    println!("npeaks: {:?}", npeaks);

    Ok(())
}
