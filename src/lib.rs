extern crate nalgebra as na;

pub mod integration;
pub mod newton;
pub mod polyfit;
pub mod quadrature;

#[allow(non_camel_case_types)]
pub type fx = f64;
