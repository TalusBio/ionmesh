use rerun;

pub trait RerunPlottable {
    fn plot(
        &self,
        rec: &mut rerun::RecordingStream,
        entry_path: String,
        log_time_in_seconds: Option<f32>,
    ) -> Result<(), Box<dyn std::error::Error>>;
}

// #[cfg(feature='viz')]
pub fn setup_recorder() -> rerun::RecordingStream {
    let rec = rerun::RecordingStreamBuilder::new("rerun_jspp_denoiser").connect();

    return rec.unwrap();
}
