
use crate::ms::frames::TimsPeak;
use crate::space::space_generics::NDPointConverter;
use crate::space::space_generics::NDPoint;

// https://github.com/rust-lang/rust/issues/35121
// The never type is not stable yet....
pub struct BypassDenseFrameBackConverter {}

impl NDPointConverter<TimsPeak, 2> for BypassDenseFrameBackConverter {
    fn convert(&self, _elem: &TimsPeak) -> NDPoint<2> {
        panic!("This should never be called")
    }
}

pub struct DenseFrameConverter {
    pub mz_scaling: f64,
    pub ims_scaling: f32,
}

impl NDPointConverter<TimsPeak, 2> for DenseFrameConverter {
    fn convert(&self, elem: &TimsPeak) -> NDPoint<2> {
        NDPoint {
            values: [
                (elem.mz / self.mz_scaling) as f32,
                (elem.mobility / self.ims_scaling),
            ],
        }
    }
}
