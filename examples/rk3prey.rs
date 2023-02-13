extern crate nalgebra as na;

use numerics::{integration::ErkIntegrator, polyfit::polyfit};

#[allow(non_camel_case_types)]
type fx = f64;

fn main() {
  #[rustfmt::skip]
  let a = na::DMatrix::from_row_slice(
    3, 3,
    &[
      0.0, 0.0, 0.0,
      1.0 / 3.0, 0.0, 0.0,
      0.0, 2.0 / 3.0, 0.0
    ],
  );
  #[rustfmt::skip]
  let b = na::DVector::from_vec(vec![
    1.0 / 4.0, 0.0, 3.0 / 4.0
  ]);
  let integrator = ErkIntegrator::new(a, b);

  let alpha1 = 3.0;
  let alpha2 = 2.0;
  let beta1 = 0.1;
  let beta2 = 0.1;

  let f = |y: &na::DVector<fx>| -> na::DVector<fx> {
    na::DVector::from_vec(vec![
      y[0] * (alpha1 - beta1 * y[1]),
      y[1] * (beta2 * y[0] - alpha2),
    ])
  };

  let y0 = na::DVector::from_vec(vec![100.0, 5.0]);
  let t = 10.0;

  let jstart = 7;
  let jend = 14;
  let jcount = jend - jstart + 1;

  let yref = na::DVector::from_vec(vec![0.319465882659820, 9.730809352326228]);

  let mut ns = na::DVector::zeros(jcount);
  let mut errors = na::DVector::zeros(jcount);

  for i in 0..jcount {
    let j = jstart + i;
    let n = 2usize.pow(j as u32);

    let y = integrator
      .solve(f, y0.clone(), t, n)
      .last()
      .unwrap()
      .clone();

    let error = (y - &yref).norm();
    println!("Step N={} Error E={}", n, error);
    ns[i] = n as fx;
    errors[i] = error;
  }
  let conv_rate = -polyfit(&ns.map(|e| e.log2()), errors.map(|e| e.log2()), 1)[0];
  println!("convergence rate r={}", conv_rate);
}
