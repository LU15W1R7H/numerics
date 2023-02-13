extern crate nalgebra as na;

use numerics::newton::newton_damped_solve;

#[allow(non_camel_case_types)]
type fx = f64;

fn main() {
  let f = |x: &na::DVector<fx>| -> na::DVector<fx> { x * x - na::DVector::from_element(1, 3.0) };
  let jf = |x: &na::DVector<fx>| -> na::DMatrix<fx> { na::DMatrix::from_element(1, 1, 2.0 * x[0]) };
  let x0 = na::DVector::from_element(1, 2.0);
  let x = newton_damped_solve(f, jf, x0, None, None).unwrap();
  println!("x: {}", x);
}
