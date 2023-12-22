// OrderedDict([
//     ('psm_id', Int64),
//     ('fragment_type', Utf8),
//     ('fragment_ordinals', Int32),
//     ('fragment_charge', Int32),
//     ('fragment_mz_calculated', Float32),
//     ('fragment_intensity', Float32)])
//

use apache_avro::{Codec, Error, Schema, Writer};
mod quad;

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
        println!("serialized = {}", serialized);
        let deserialized: Parent = serde_json::from_str(&serialized).unwrap();
        assert_eq!(extract_parent, deserialized);
    }
}

fn quad_main() {
    let boundary = quad::Boundary {
        x_center: 0.0,
        y_center: 0.0,
        width: 50.0,
        height: 50.0,
    };
    let radius = 5.0;
    let mut quad_tree = quad::RadiusQuadTree::new(boundary, 2, radius);

    // Insert some points clustered close to 25,25
    quad_tree.insert(quad::Point { x: 20.0, y: 20.0 }, &"B");
    quad_tree.insert(quad::Point { x: 5.45, y: 4.29 }, &"A");
    quad_tree.insert(quad::Point { x: 2.69, y: 9.25 }, &"A");
    quad_tree.insert(quad::Point { x: 12.94, y: 18.66 }, &"A");
    quad_tree.insert(quad::Point { x: 18.13, y: 0.05 }, &"A");
    quad_tree.insert(quad::Point { x: 2.50, y: 14.66 }, &"A");
    quad_tree.insert(quad::Point { x: 17.62, y: 15.26 }, &"A");
    quad_tree.insert(quad::Point { x: 7.14, y: 14.54 }, &"A");
    quad_tree.insert(quad::Point { x: 1.10, y: 4.22 }, &"A");
    quad_tree.insert(quad::Point { x: 13.28, y: 14.35 }, &"A");
    quad_tree.insert(quad::Point { x: 14.06, y: 13.29 }, &"A");

    // Query points within the radius
    let query_point = quad::Point { x: 5.0, y: 5.0 };

    let mut result = Vec::new();
    quad_tree.query(query_point, &mut result);
    let qt_json = quad_tree.to_json();

    // write to disk
    std::fs::write("quad_tree.json", qt_json).unwrap();
}

fn main() -> Result<(), Error> {
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
    println!("serialized = {}", serialized);

    let schema = Schema::parse_str(RAW_SCHEMA)?;
    let mut writer = Writer::with_codec(&schema, Vec::new(), Codec::Deflate);
    writer.append_ser(extract_parent)?;

    quad_main();
    Ok(())
}
