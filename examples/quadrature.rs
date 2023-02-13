use numerics::quadrature::{self, QuadRule};

fn main() {
  let exact = 2.0;

  let mut prev_error = f64::INFINITY;
  for i in 1..=10 {
    let fi = i as f64;

    let qr = QuadRule::new_gauss(i);
    let value = quadrature::evalquad(f64::sin, 0.0, std::f64::consts::PI, &qr);
    let error = (exact - value).abs();
    //let order = prev_error / error;
    let order = -(fi / (fi - 1.0)).log2() / (error / prev_error).log2();
    prev_error = error;

    println!("value: {}, error: {}, order: {}", value, error, order);
  }
}
