#![allow(dead_code)]
use std::sync::Arc;
use std::usize;

const INVALID_VARIABLE: usize = usize::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Node {
    Leaf { value: bool },
    Branch {
        variable: usize,
        low: usize,
        high: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Branch {
    pub variable: usize,
    pub low: usize,
    pub high: usize,
}

#[derive(Debug, Clone)]
pub struct Graph {
    nodes: Vec<NodeInner>,
    variable_count: usize,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: vec![NodeInner::zero(), NodeInner::one()],
            variable_count: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut nodes = Vec::with_capacity(capacity);
        nodes.push(NodeInner::zero());
        nodes.push(NodeInner::one());
        Graph {
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

    pub fn expand(&self, index: usize) -> Node {
        let node = self.nodes[index];
        match node.as_literal() {
            Some(value) => Node::Leaf { value },
            None => {
                Node::Branch {
                    variable: node.variable,
                    low: (index as isize + node.low) as usize,
                    high: (index as isize + node.high) as usize,
                }
            }
        }
    }

    pub fn expand_branch(&self, index: usize) -> Branch {
        if let Node::Branch {
                   variable,
                   low,
                   high,
               } = self.expand(index) {
            Branch {
                variable,
                low,
                high,
            }
        } else {
            panic!("expand_branch called on leaf");
        }
    }
}

#[derive(Clone, Debug)]
pub struct OrderedDiagram {
    pub root: usize,
    pub order: Arc<Vec<usize>>,
}

impl OrderedDiagram {
    pub fn from_root(root: usize, variable_count: usize) -> Self {
        OrderedDiagram {
            root,
            order: Arc::new((0..variable_count).collect()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_zero() {
        let mut g = Graph::new();
        let _zero = g.zero();
    }

    #[test]
    fn construct_one() {
        let mut g = Graph::new();
        let _one = g.one();
    }

    #[test]
    fn construct_identity() {
        let mut g = Graph::new();
        let zero = g.zero();
        let one = g.one();
        let _branch = g.branch(0, zero, one);
    }
}
