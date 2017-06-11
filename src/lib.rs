#![allow(dead_code)]
#[macro_use]
extern crate log;
extern crate bit_vec;
extern crate smallvec;
extern crate vec_map;
extern crate rand;
extern crate env_logger;
extern crate dot;

use bit_vec::BitVec;

mod diagram;
mod evolve;
mod walk;
mod random;
mod render;

use diagram::{Graph, Node};

fn evaluate_diagram(graph: &Graph, root: usize, input: &BitVec) -> bool {
    let mut node = root;
    loop {
        match graph.expand(node) {
            Node::Leaf { value } => {
                return value;
            }
            Node::Branch {
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
        let mut g = Graph::new();
        let zero = g.zero();
        assert_eq!(evaluate_diagram(&g, zero, &BitVec::from_elem(0, false)),
                   false);
    }

    #[test]
    fn evaluate_one() {
        let mut g = Graph::new();
        let one = g.one();
        assert_eq!(evaluate_diagram(&g, one, &BitVec::from_elem(0, false)),
                   true);
    }

    #[test]
    fn evaluate_identity() {
        let mut g = Graph::new();
        let zero = g.zero();
        let one = g.one();
        let branch = g.branch(0, zero, one);
        assert_eq!(evaluate_diagram(&g, branch, &BitVec::from_elem(1, false)),
                   false);
        assert_eq!(evaluate_diagram(&g, branch, &BitVec::from_elem(1, true)),
                   true);
    }
}
