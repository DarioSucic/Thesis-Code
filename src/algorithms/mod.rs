mod bandit_pam;
mod pam;

pub use self::bandit_pam::*;
pub use self::pam::*;

use crate::{measure::Measurable, Float};

pub trait Solver<T: Float> {
    fn fit(d: &impl Measurable<T>, k: usize) -> Vec<usize>;
}
