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
mod mod_types;
mod ms;
mod ms_denoise;
mod quad;
mod space_generics;
mod tdf;

extern crate pretty_env_logger;
#[macro_use]
extern crate log;

// https://github.com/lerouxrgd/rsgen-avro
// This can be used to generate the structs from
// the avro schema.
const RAW_SCHEMA: &str = r#"
{
    "type": "record",
    "name": "Parent",
    "fields": [
        { "name": "charge", "type": "int" },
        { "name": "id", "type": "int" },
        { "name": "mobility", "type": "float" },
        { "name": "mz", "type": "double" },
        { "name": "retention_time", "type": "float" },
        {
            "name": "peaks",
            "type": { 
                "type": "array",
                "items": [
                    "null",
                    {
                        "type": "record",
                        "name": "Peak",
                        "fields": [
                            {
                                "name": "charge",
                                "type": [
                                    "null",
                                    "int"
                                ],
                                "default": null
                            },
                            { "name": "ion_id", "type": "int" },
                            { "name": "mz", "type": "double" },
                            { "name": "parent_id", "type": "int" }
                        ]
                    }
                ]
            }
        }
    ]
}
"#;

#[derive(Debug, PartialEq, Clone, serde::Deserialize, serde::Serialize)]
pub struct Peak {
    #[serde(default = "default_peak_charge")]
    pub charge: Option<i32>,
    pub ion_id: i32,
    pub mz: f64,
    pub parent_id: i32,
}

#[inline(always)]
fn default_peak_charge() -> Option<i32> {
    None
}

#[derive(Debug, PartialEq, Clone, serde::Deserialize, serde::Serialize)]
pub struct Parent {
    pub charge: i32,
    pub id: i32,
    pub mobility: f32,
    pub mz: f64,
    pub retention_time: f32,
    pub peaks: Vec<Option<Peak>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serde() {
        let ion = Peak {
            ion_id: 0,
            charge: Some(1),
            mz: 123.123f64,
            parent_id: 0,
        };
        let extract_ions = vec![Some(ion)];
        let extract_parent = Parent {
            id: 0,
            charge: 0,
            retention_time: 123.123f32,
            mobility: 1.21f32,
            mz: 123.123f64,
            peaks: extract_ions,
        };

        let serialized = serde_json::to_string(&extract_parent).unwrap();
        info!("serialized = {}", serialized);
        let deserialized: Parent = serde_json::from_str(&serialized).unwrap();
        assert_eq!(extract_parent, deserialized);
    }
}

fn main() -> Result<(), Error> {
    pretty_env_logger::init();

    let path_use = String::from("/Users/sebastianpaez/git/2023_dev_diadem_report/data/231121_RH30_NMIAA_E3_DIA_S2-B3_1_5353.d");
    let dia_info = tdf::read_dia_frame_info(path_use.clone());
    // ms_denoise::read_all_ms1_denoising(path_use.clone());
    let mut dia_frames = ms_denoise::read_all_dia_denoising(path_use.clone());

    Ok(())
}
