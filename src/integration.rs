use crate::{fx, newton::newton_damped_solve};

/// explicit runge-kutta integrator
pub struct ErkIntegrator {
  // A in butcher tableau
  a: na::DMatrix<fx>,
  // b in butcher tableau
  b: na::DVector<fx>,
  // number of stages
  s: usize,
}

impl ErkIntegrator {
  pub fn new(a: na::DMatrix<fx>, b: na::DVector<fx>) -> Self {
    let s = b.len();
    assert!(a.ncols() == s && a.nrows() == s);
    Self { a, b, s }
  }

  pub fn step<F>(&self, f: F, y0: &na::DVector<fx>, h: fx) -> na::DVector<fx>
  where
    F: Fn(&na::DVector<fx>) -> na::DVector<fx>,
  {
    let d = y0.len();

    let mut ks = Vec::with_capacity(self.s);
    for i in 0..self.s {
      let mut sum_a = na::DVector::zeros(d);
      for j in 0..i {
        sum_a += self.a[(i, j)] * &ks[j];
      }
      ks.push(f(&(y0 + h * sum_a)));
    }

    let mut sum_b = na::DVector::zeros(d);
    for i in 0..self.s {
      sum_b += self.b[i] * &ks[i];
    }
    y0 + h * sum_b
  }

  pub fn solve<F>(&self, f: F, y0: na::DVector<fx>, t: fx, n: usize) -> Vec<na::DVector<fx>>
  where
    F: Fn(&na::DVector<fx>) -> na::DVector<fx>,
  {
    let h = t / n as fx;

    std::iter::successors(Some(y0), |yp| Some(self.step(&f, yp, h)))
      .take(n + 1)
      .collect()
  }
}

/// implicit runge-kutta integrator
pub struct IrkIntegrator {
  // A in butcher tableau
  a: na::DMatrix<fx>,
  // b in butcher tableau
  b: na::DVector<fx>,
  // number of stages
  s: usize,
}

impl IrkIntegrator {
  pub fn new(a: na::DMatrix<fx>, b: na::DVector<fx>) -> Self {
    let s = b.len();
    assert!(a.ncols() == s && a.nrows() == s);
    Self { a, b, s }
  }

  pub fn step<F, Jf>(&self, f: F, jf: Jf, y0: &na::DVector<fx>, h: fx) -> na::DVector<fx>
  where
    F: Fn(&na::DVector<fx>) -> na::DVector<fx>,
    Jf: Fn(&na::DVector<fx>) -> na::DMatrix<fx>,
  {
    let d = y0.len();
    let s = self.s;
    let a = &self.a;
    let b = &self.b;

    let gv = {
      let ff = |gv: &na::DVector<fx>| -> na::DVector<fx> {
        let mut r = gv.clone();
        for i in 0..s {
          r -=
            h * a.column(i).kronecker(&na::DMatrix::identity(d, d)) * f(&(y0 + gv.rows(i * d, d)));
        }
        r
      };
      let jff = |gv: &na::DVector<fx>| -> na::DMatrix<fx> {
        let mut r = na::DMatrix::zeros(s * d, s * d);
        for i in 0..s {
          r.view_mut((i * d, 0), (d, s * d))
            .copy_from(&a.row(i).kronecker(&jf(&(y0 + gv.rows(i * d, d)))));
        }
        r = na::DMatrix::identity(s * d, s * d) - h * r;
        r
      };

      newton_damped_solve(ff, jff, na::DVector::zeros(s * d), None, None).unwrap()
    };

    let mut ks = na::DMatrix::zeros(d, s);
    for i in 0..s {
      ks.column_mut(i).copy_from(&f(&(y0 + gv.rows(i * d, d))));
    }

    y0 + h * ks * b
  }

  pub fn solve<F, Jf>(
    &self,
    f: F,
    jf: Jf,
    y0: na::DVector<fx>,
    t: fx,
    n: usize,
  ) -> Vec<na::DVector<fx>>
  where
    F: Fn(&na::DVector<fx>) -> na::DVector<fx>,
    Jf: Fn(&na::DVector<fx>) -> na::DMatrix<fx>,
  {
    let h = t / n as fx;

    std::iter::successors(Some(y0), |yp| Some(self.step(&f, &jf, yp, h)))
      .take(n + 1)
      .collect()
  }
}
