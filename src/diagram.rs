#![allow(dead_code)]
use std::usize;

const INVALID_VARIABLE: usize = usize::MAX;

#[derive(Clone, Copy, PartialEq, Eq)]
struct NodeInner {
    variable: usize,
    low: isize,
    high: isize,
}

impl NodeInner {
    fn as_literal(&self) -> Option<bool> {
        if self.variable == INVALID_VARIABLE {
            Some(self.low == 1)
        } else {
            None
        }
    }

    fn zero() -> Self {
        NodeInner {
            variable: INVALID_VARIABLE,
            low: 0,
            high: 0,
        }
    }

    fn one() -> Self {
        NodeInner {
            variable: INVALID_VARIABLE,
            low: 1,
            high: 1,
        }
    }
}

pub enum NodeX {
    Leaf { value: bool },
    Branch {
        variable: usize,
        low: usize,
        high: usize,
    },
}

pub struct Diagram {
    nodes: Vec<NodeInner>,
    variable_count: usize,
}

impl Diagram {
    pub fn new() -> Self {
        Diagram {
            nodes: vec![NodeInner::zero(), NodeInner::one()],
            variable_count: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut nodes = Vec::with_capacity(capacity);
        nodes.push(NodeInner::zero());
        nodes.push(NodeInner::one());
        Diagram {
            nodes: nodes,
            variable_count: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn zero(&mut self) -> usize {
        0
    }

    pub fn one(&mut self) -> usize {
        1
    }

    pub fn branch(&mut self, variable: usize, low: usize, high: usize) -> usize {
        if variable + 1 > self.variable_count {
            self.variable_count = variable + 1;
        }
        let index = self.nodes.len() as isize;
        let node = NodeInner {
            variable,
            low: low as isize - index,
            high: high as isize - index,
        };
        self.nodes.push(node);
        return index as usize;
    }

    pub fn expand(&self, index: usize) -> NodeX {
        let node = self.nodes[index];
        match node.as_literal() {
            Some(value) => NodeX::Leaf { value },
            None => {
                NodeX::Branch {
                    variable: node.variable,
                    low: (index as isize + node.low) as usize,
                    high: (index as isize + node.high) as usize,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_zero() {
        let mut d = Diagram::new();
        let _zero = d.zero();
    }

    #[test]
    fn construct_one() {
        let mut d = Diagram::new();
        let _one = d.one();
    }

    #[test]
    fn construct_identity() {
        let mut d = Diagram::new();
        let zero = d.zero();
        let one = d.one();
        let _branch = d.branch(0, zero, one);
    }
}
