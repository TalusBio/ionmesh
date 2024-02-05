pub trait RerunPlottable<T> {
    fn plot(
        &self,
        rec: &mut rerun::RecordingStream,
        entry_path: String,
        log_time_in_seconds: Option<f32>,
        required_extras: T,
    ) -> Result<(), Box<dyn std::error::Error>>;
}

// #[cfg(feature='viz')]
pub fn setup_recorder() -> rerun::RecordingStream {
    let rec = rerun::RecordingStreamBuilder::new("rerun_jspp_denoiser").connect();
    match rec {
        Ok(rec) => {
            rec.set_time_seconds("rt_seconds", 0.0f64);
            rec
        }
        Err(e) => {
            // If the viz mode is on ... there has to be a viz...
            panic!("Error setting up recorder: {:?}", e);
        }
    }
}
