use crate::fx;

// Solver for polynomial linear least squares data fitting problem
// data points passed in t and y, order = degree + 1
pub fn polyfit(t: &na::DVector<fx>, y: na::DVector<fx>, order: usize) -> na::DVector<fx> {
  let mut a = na::DMatrix::from_element(t.len(), order + 1, 1.0);
  for i in 1..(order + 1) {
    a.set_column(i, &a.column(i - 1).component_mul(t));
  }
  let eps = (a.nrows() * a.ncols()) as fx * a.norm() * fx::EPSILON;
  let coeffs = a.svd(true, true).solve(&y, eps).unwrap();
  na::DVector::from_iterator(coeffs.len(), coeffs.into_iter().copied().rev())
}
