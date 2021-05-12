mod bandit_pam;
mod pam;

pub use self::bandit_pam::*;
pub use self::pam::*;

use crate::{measure::Measurable, BoundedNum};

pub trait Solver<T: BoundedNum> {
    fn fit(d: &impl Measurable<T>, n: usize) -> Vec<usize>;
}
