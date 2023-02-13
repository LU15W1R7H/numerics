use crate::fx;

pub struct QuadRule {
  pub nodes: na::DVector<fx>,
  pub weights: na::DVector<fx>,
}

impl QuadRule {
  pub fn new_gauss(n: usize) -> Self {
    let mut m = na::DMatrix::zeros(n, n);
    for i in 1..n {
      let fi = i as fx;
      let b = fi / (4.0 * fi * fi - 1.0).sqrt();
      m[(i, i - 1)] = b;
      m[(i - 1, i)] = b;
    }

    let eigen = m.symmetric_eigen();
    let nodes = eigen.eigenvalues;
    let weights = 2.0 * eigen.eigenvectors.row(0).transpose().map(|e| e.powf(2.0));

    Self { nodes, weights }
  }

  pub fn nodes_scaled(&self, a: fx, b: fx) -> na::DVector<fx> {
    0.5 * self.nodes.map(|e| 1.0 - e) * a + 0.5 * self.nodes.map(|e| 1.0 + e) * b
  }
  pub fn weights_scaled(&self, a: fx, b: fx) -> na::DVector<fx> {
    0.5 * (b - a) * &self.weights
  }
}

pub fn evalquad<F>(f: F, a: fx, b: fx, qr: &QuadRule) -> fx
where
  F: Fn(fx) -> fx,
{
  let nodes = qr.nodes_scaled(a, b);
  let weights = qr.weights_scaled(a, b);
  weights.dot(&nodes.map(f))
}

pub fn evalquad2drect<F>(f: F, a: fx, b: fx, c: fx, d: fx, qr: &QuadRule) -> fx
where
  F: Fn(fx, fx) -> fx,
{
  let fx = |x| {
    let fy = |y| f(x, y);
    evalquad(fy, c, d, qr)
  };
  evalquad(fx, a, b, qr)
}
