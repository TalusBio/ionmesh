// Floating-point precision is configured here
// https://users.rust-lang.org/t/generics-using-either-f32-or-f64/28647/3
#[cfg(not(feature = "f64"))]
pub type Float = f32;
#[cfg(not(feature = "f64"))]
pub use std::f32 as floats;
#[cfg(feature = "f64")]
pub type Float = f64;
#[cfg(feature = "f64")]
pub use std::f64 as floats;
