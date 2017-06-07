#![allow(dead_code)]
extern crate bit_vec;
extern crate smallvec;
extern crate vec_map;
extern crate rand;

use bit_vec::BitVec;

mod diagram;
mod evolve;

use diagram::{Diagram, NodeX};

fn evaluate_diagram(diagram: &Diagram, root: usize, input: &BitVec) -> bool {
    let mut node = root;
    loop {
        match diagram.expand(node) {
            NodeX::Leaf { value } => {
                return value;
            }
            NodeX::Branch {
                variable,
                low,
                high,
            } => {
                if input[variable] {
                    node = high;
                } else {
                    node = low;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_zero() {
        let mut d = Diagram::new();
        let zero = d.zero();
        assert_eq!(evaluate_diagram(&d, zero, &BitVec::from_elem(0, false)),
                   false);
    }

    #[test]
    fn evaluate_one() {
        let mut d = Diagram::new();
        let one = d.one();
        assert_eq!(evaluate_diagram(&d, one, &BitVec::from_elem(0, false)),
                   true);
    }

    #[test]
    fn evaluate_identity() {
        let mut d = Diagram::new();
        let zero = d.zero();
        let one = d.one();
        let branch = d.branch(0, zero, one);
        assert_eq!(evaluate_diagram(&d, branch, &BitVec::from_elem(1, false)),
                   false);
        assert_eq!(evaluate_diagram(&d, branch, &BitVec::from_elem(1, true)),
                   true);
    }
}
