use crate::fx;

pub fn newton_step<F, Jf>(f: F, jf: Jf, x: na::DVector<fx>) -> na::DVector<fx>
where
  F: Fn(&na::DVector<fx>) -> na::DVector<fx>,
  Jf: Fn(&na::DVector<fx>) -> na::DMatrix<fx>,
{
  let delta = jf(&x).lu().solve(&f(&x)).unwrap();
  x - delta
}

pub fn newton_step_scalar<F, Jf>(f: F, jf: Jf, x: fx) -> fx
where
  F: Fn(fx) -> fx,
  Jf: Fn(fx) -> fx,
{
  let delta = jf(x).recip() * f(x);
  x - delta
}

pub fn newton_solve<F, Jf>(
  f: F,
  jf: Jf,
  mut x: na::DVector<fx>,
  rtol: Option<fx>,
  atol: Option<fx>,
  max_iterations: Option<usize>,
) -> Option<na::DVector<fx>>
where
  F: Fn(&na::DVector<fx>) -> na::DVector<fx>,
  Jf: Fn(&na::DVector<fx>) -> na::DMatrix<fx>,
{
  let rtol = rtol.unwrap_or(1e-4);
  let atol = atol.unwrap_or(1e-6);
  let max_iterations = max_iterations.unwrap_or(10_000);

  for _ in 0..max_iterations {
    let jf_lu = jf(&x).lu();
    let delta = jf_lu.solve(&f(&x)).unwrap();
    let delta_norm = delta.norm();
    x -= delta;

    if delta_norm < atol || delta_norm < rtol * x.norm() {
      return Some(x);
    }
  }
  None
}

pub fn newton_solve_scalar<F, Jf>(
  f: F,
  jf: Jf,
  mut x: fx,
  rtol: Option<fx>,
  atol: Option<fx>,
  max_iterations: Option<usize>,
) -> Option<fx>
where
  F: Fn(fx) -> fx,
  Jf: Fn(fx) -> fx,
{
  let rtol = rtol.unwrap_or(1e-4);
  let atol = atol.unwrap_or(1e-6);
  let max_iterations = max_iterations.unwrap_or(10_000);

  for _ in 0..max_iterations {
    let delta = jf(x).recip() * f(x);
    let delta_norm = delta.abs();
    x -= delta;

    if delta_norm < atol || delta_norm < rtol * x.abs() {
      return Some(x);
    }
  }
  None
}

pub fn newton_damped_solve<F, Jf>(
  f: F,
  jf: Jf,
  mut x: na::DVector<fx>,
  rtol: Option<fx>,
  atol: Option<fx>,
) -> Option<na::DVector<fx>>
where
  F: Fn(&na::DVector<fx>) -> na::DVector<fx>,
  Jf: Fn(&na::DVector<fx>) -> na::DMatrix<fx>,
{
  let rtol = rtol.unwrap_or(1e-4);
  let atol = atol.unwrap_or(1e-6);
  let lambda_min = 1e-3;

  let mut lambda = 1.0;
  loop {
    let jf_lu = jf(&x).lu();
    let delta = jf_lu.solve(&f(&x)).unwrap();
    let delta_norm = delta.norm();

    let mut x_candidate;
    let mut delta_norm_candidate;
    lambda *= 2.0;
    loop {
      lambda /= 2.0;
      if lambda < lambda_min {
        return None;
      }
      x_candidate = &x - lambda * &delta;
      let delta_candidate = jf_lu.solve(&(f(&x_candidate))).unwrap();
      delta_norm_candidate = delta_candidate.norm();

      if delta_norm_candidate < (1.0 - lambda / 2.0) * delta_norm {
        break;
      }
    }
    x = x_candidate;
    lambda = (2.0 * lambda).min(1.0);

    if delta_norm_candidate < rtol * x.norm() || delta_norm_candidate < atol {
      break;
    };
  }
  Some(x)
}
