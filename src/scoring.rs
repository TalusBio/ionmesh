use std::fs::read_to_string;
use std::str::FromStr;

use crate::tracing::PseudoSpectrum;
use indicatif::ParallelProgressIterator;
use log::warn;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use serde::Deserialize;
use serde::Serialize;
use serde::ser::SerializeStruct;
use sage_core::database::Builder as SageDatabaseBuilder;
use sage_core::database::Parameters as SageDatabaseParameters;
use sage_core::database::{EnzymeBuilder, IndexedDatabase};
use sage_core::ion_series::Kind;
use sage_core::mass::Tolerance;
use sage_core::ml::linear_discriminant::score_psms;
use sage_core::modification::ModificationSpecificity;
use sage_core::scoring::Scorer;
use sage_core::spectrum::{Precursor, RawSpectrum, Representation, SpectrumProcessor};
use sage_core::scoring::Feature;
use serde::Serializer;

use std::collections::HashMap;
use std::error::Error;
use std::fs;

use rayon::prelude::*;

// Copied from sage_cloudpath just to prevent one more depencency ...

pub fn read_fasta<S>(
    path: S,
    decoy_tag: S,
    generate_decoys: bool,
) -> Result<sage_core::fasta::Fasta, Box<dyn Error>>
where
    S: AsRef<str>,
{
    let contents = fs::read_to_string(path.as_ref())?;
    Ok(sage_core::fasta::Fasta::parse(
        contents,
        decoy_tag.as_ref(),
        generate_decoys,
    ))
}

#[derive(Debug, Clone)]
struct SerializableFeature<'a> {
    peptide: String,
    feature: &'a Feature,
}

impl <'a>SerializableFeature<'a> {
    fn from_feature(feat: &'a sage_core::scoring::Feature, db: &IndexedDatabase) -> Self {
        let peptide = db[feat.peptide_idx].to_string().clone();
        SerializableFeature { peptide, feature: feat }
    }
}



impl Serialize for SerializableFeature<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut row = serializer.serialize_struct("SerializableFeature", 3)?;
        row.serialize_field("peptide", &self.peptide)?;
        row.serialize_field("psm_id", &self.feature.psm_id)?;
        row.serialize_field("peptide_len", &self.feature.peptide_len)?;
        row.serialize_field("spec_id", &self.feature.spec_id)?;
        row.serialize_field("file_id", &self.feature.file_id)?;
        row.serialize_field("rank", &self.feature.rank)?;
        row.serialize_field("label", &self.feature.label)?;
        row.serialize_field("expmass", &self.feature.expmass)?;
        row.serialize_field("calcmass", &self.feature.calcmass)?;
        row.serialize_field("charge", &self.feature.charge)?;
        row.serialize_field("rt", &self.feature.rt)?;
        row.serialize_field("aligned_rt", &self.feature.aligned_rt)?;
        row.serialize_field("predicted_rt", &self.feature.predicted_rt)?;
        row.serialize_field("delta_rt_model", &self.feature.delta_rt_model)?;
        row.serialize_field("delta_mass", &self.feature.delta_mass)?;
        row.serialize_field("isotope_error", &self.feature.isotope_error)?;
        row.serialize_field("average_ppm", &self.feature.average_ppm)?;
        row.serialize_field("hyperscore", &self.feature.hyperscore)?;
        row.serialize_field("delta_next", &self.feature.delta_next)?;
        row.serialize_field("delta_best", &self.feature.delta_best)?;
        row.serialize_field("matched_peaks", &self.feature.matched_peaks)?;
        row.serialize_field("longest_b", &self.feature.longest_b)?;
        row.serialize_field("longest_y", &self.feature.longest_y)?;
        row.serialize_field("longest_y_pct", &self.feature.longest_y_pct)?;
        row.serialize_field("missed_cleavages", &self.feature.missed_cleavages)?;
        row.serialize_field("matched_intensity_pct", &self.feature.matched_intensity_pct)?;
        row.serialize_field("scored_candidates", &self.feature.scored_candidates)?;
        row.serialize_field("poisson", &self.feature.poisson)?;
        row.serialize_field("discriminant_score", &self.feature.discriminant_score)?;
        row.serialize_field("posterior_error", &self.feature.posterior_error)?;
        row.serialize_field("spectrum_q", &self.feature.spectrum_q)?;
        row.serialize_field("peptide_q", &self.feature.peptide_q)?;
        row.serialize_field("protein_q", &self.feature.protein_q)?;
        row.serialize_field("ms2_intensity", &self.feature.ms2_intensity)?;
        row.end()
    }
}


//

fn pseudospectrum_to_spec(pseudo: PseudoSpectrum, scan_id: String) -> RawSpectrum {
    let file_id = 1;
    let ms_level = 2;

    let prec_center = (pseudo.quad_low + pseudo.quad_high) / 2.;
    let prec_width = pseudo.quad_high - pseudo.quad_low;

    let precursor = Precursor {
        mz: prec_center as f32,
        intensity: None,
        charge: None,
        spectrum_ref: None,
        isolation_window: Some(Tolerance::Da(
            (-prec_width / 2.) as f32,
            (prec_width / 2.) as f32,
        )),
    };

    let (mzs, ints): (Vec<f32>, Vec<f32>) = pseudo
        .peaks
        .into_iter()
        .map(|x| (x.0 as f32, x.1 as f32))
        .unzip();
    let tic = ints.iter().sum();

    let spec = RawSpectrum {
        file_id,
        ms_level,
        id: scan_id,
        precursors: vec![precursor],
        representation: Representation::Centroid,
        scan_start_time: pseudo.rt as f32,
        mz: mzs,
        intensity: ints,
        ion_injection_time: 100.,
        total_ion_current: tic,
    };

    spec
}

pub fn score_pseudospectra(
    elems: Vec<PseudoSpectrum>,
    fasta_path: String,
) -> Result<(), Box<dyn Error>> {
    // 1. Buid raw spectra from the pseudospectra

    let take_top_n = 250;
    let min_fragment_mz = 250.;
    let max_fragment_mz = 2000.;
    let deisotope = true;

    let specs = elems
        .into_par_iter()
        .enumerate()
        .map(|(i, x)| {
            let scan_id = format!("{}_{}", i, "test");
            pseudospectrum_to_spec(x, scan_id)
        })
        .collect::<Vec<_>>();
    let procesor = SpectrumProcessor::new(take_top_n, min_fragment_mz, max_fragment_mz, deisotope);
    let spectra = specs
        .into_par_iter()
        .map(|x| procesor.process(x))
        .collect::<Vec<_>>();

    // Parameters -> Parameters::build -> IndexedDb
    let mut static_mods: HashMap<ModificationSpecificity, f32> = HashMap::new();
    static_mods.insert(ModificationSpecificity::from_str("C").unwrap(), 57.02146);

    let mut variable_mods: HashMap<ModificationSpecificity, Vec<f32>> = HashMap::new();
    variable_mods.insert(
        ModificationSpecificity::from_str("M").unwrap(),
        vec![15.99491],
    );

    let parameters = SageDatabaseParameters {
        bucket_size: 8192,
        enzyme: EnzymeBuilder {
            missed_cleavages: Some(2),
            min_len: Some(6),
            max_len: Some(35),
            cleave_at: Some("KR".into()),
            restrict: None,
            semi_enzymatic: None,
            c_terminal: None,
        },
        fragment_min_mz: min_fragment_mz,
        fragment_max_mz: max_fragment_mz,
        peptide_min_mass: 500.,
        peptide_max_mass: 3500.,
        ion_kinds: vec![Kind::B, Kind::Y],
        min_ion_index: 2,
        static_mods: (static_mods),
        variable_mods: (variable_mods),
        max_variable_mods: 1,
        decoy_tag: "rev_".into(),
        generate_decoys: true,
        fasta: fasta_path.clone(),
    };

    let sage_fasta = read_fasta(
        fasta_path.clone(),
        parameters.decoy_tag.clone(),
        parameters.generate_decoys,
    )
    .expect("Error reading fasta");

    let db = parameters.clone().build(sage_fasta);

    let precursor_tolerance = Tolerance::Ppm(-15., 15.);
    let scorer = Scorer {
        db: &db,
        precursor_tol: precursor_tolerance,
        fragment_tol: Tolerance::Ppm(-10., 10.),
        min_matched_peaks: 3,
        min_isotope_err: 0,
        max_isotope_err: 0,
        min_precursor_charge: 2,
        max_precursor_charge: 4,
        max_fragment_charge: Some(1),
        min_fragment_mass: 200.,
        max_fragment_mass: 4000.,
        chimera: true,
        report_psms: 2,
        wide_window: true,
        annotate_matches: false,
    };

    let progbar = indicatif::ProgressBar::new(spectra.len() as u64);

    let mut features = spectra
        .par_iter()
        .progress_with(progbar)
        .flat_map(|spec| scorer.score(spec))
        .collect::<Vec<_>>();

    let discriminant = score_psms(&mut features, precursor_tolerance);
    if discriminant.is_none() {
        // Stolen from sage ...
        log::warn!("linear model fitting failed, falling back to heuristic discriminant score");
        features.par_iter_mut().for_each(|feat| {
            feat.discriminant_score = (-feat.poisson as f32).ln_1p() + feat.longest_y_pct / 3.0
        });
    }

    features.par_sort_unstable_by(|a, b| b.discriminant_score.total_cmp(&a.discriminant_score));
    let num_q_001 = sage_core::ml::qvalue::spectrum_q_value(&mut features);
    let q_peptide = sage_core::fdr::picked_peptide(&db, &mut features);
    let q_protein = sage_core::fdr::picked_protein(&db, &mut features);

    // Serialize to a csv for debugging
    warn!("Writing features to features.csv ... and sebastian should delete this b4 publishing...");
    let mut wtr = csv::Writer::from_path("features.csv")?;
    for feat in &features {
        let s_feat = SerializableFeature::from_feature(feat, &db);
        wtr.serialize(s_feat)?;
    }
    wtr.flush()?;
    drop(wtr);

    println!("Number of psms at 0.01 FDR: {}", num_q_001);
    println!("Number of peptides at 0.01 FDR: {}", q_peptide);
    println!("Number of proteins at 0.01 FDR: {}", q_protein);

    Ok(())
}