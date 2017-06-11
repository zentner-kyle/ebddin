use diagram::{Graph, Node, OrderedDiagram};
use std::rc::Rc;

#[derive(Debug)]
pub struct PathIter<'a> {
    visit_count: Vec<u8>,
    path: Rc<Vec<usize>>,
    variables: Rc<Vec<Option<bool>>>,
    variable_set_depth: Vec<usize>,
    node: usize,
    graph: &'a Graph,
    go_up: bool,
}

impl<'a> PathIter<'a> {
    pub fn new<'d>(diagram: &'d OrderedDiagram, graph: &'a Graph) -> Self {
        let variable_count = diagram.order.len();
        let tree_height_guess = variable_count;
        PathIter {
            visit_count: Vec::with_capacity(tree_height_guess),
            path: Rc::new(Vec::with_capacity(tree_height_guess)),
            variables: Rc::new(vec![None; variable_count]),
            variable_set_depth: vec![0; variable_count],
            node: diagram.root,
            graph,
            go_up: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Path {
    path: Rc<Vec<usize>>,
    variables: Rc<Vec<Option<bool>>>,
    node: usize,
    value: Option<bool>,
}

impl Path {
    fn new(path: &Rc<Vec<usize>>,
           variables: &Rc<Vec<Option<bool>>>,
           node: usize,
           value: Option<bool>)
           -> Self {
        Path {
            path: path.clone(),
            variables: variables.clone(),
            node,
            value,
        }
    }
}

impl<'a> PathIter<'a> {
    fn go_up(&mut self) -> bool {
        if let Some(node) = Rc::make_mut(&mut self.path).pop() {
            self.node = node;
            return false;
        } else {
            return true;
        }
    }

    fn go_down(&mut self, node: usize) {
        Rc::make_mut(&mut self.path).push(self.node);
        self.node = node;
        while self.visit_count.len() > self.depth() {
            debug!("unmarking depth {}", self.visit_count.len());
            self.visit_count.pop();
        }
    }

    fn depth(&self) -> usize {
        self.path.len()
    }

    fn visited_once(&self) -> bool {
        self.visit_count.len() > self.depth()
    }

    fn visited_twice(&self) -> bool {
        self.visit_count
            .get(self.depth())
            .cloned()
            .unwrap_or(0) > 1
    }

    fn visited_thrice(&self) -> bool {
        self.visit_count
            .get(self.depth())
            .cloned()
            .unwrap_or(0) > 2
    }

    fn mark(&mut self) {
        let depth = self.depth();
        while self.visit_count.len() <= depth {
            self.visit_count.push(0);
        }
        self.visit_count[depth] += 1;
    }
}

impl<'a> Iterator for PathIter<'a> {
    type Item = Path;

    fn next(&mut self) -> Option<Self::Item> {
        debug!(">>>");
        // Algorithm overview:
        // By default, go down/low.
        // On a leaf, go up. Mark the result node as visited once and go down/left.
        // If node has already been visited once, go up again instead.
        loop {
            let depth = self.path.len();
            let visited_once = self.visited_once();
            let visited_twice = self.visited_twice();
            let visited_thrice = self.visited_thrice();
            debug!("======");
            self.mark();
            debug!("node# {}", self.node);
            if visited_once {
                debug!("visited once");
            }
            if visited_twice {
                debug!("visited twice");
            }
            if visited_thrice {
                debug!("visited thrice");
            }
            match self.graph.expand(self.node) {
                Node::Leaf { value } => {
                    debug!("leaf");
                    if visited_once {
                        debug!("already visited");
                        if self.go_up() {
                            debug!("done");
                            return None;
                        }
                    } else {
                        debug!("new leaf, output path");
                        return Some(Path::new(&self.path, &self.variables, self.node, Some(value)));
                    }
                }
                Node::Branch {
                    variable,
                    low,
                    high,
                } => {
                    if let Some(value) = self.variables[variable].clone() {
                        debug!("variable {} has value {}", variable, value);
                        let variable_set_here = self.variable_set_depth[variable] == depth;
                        if variable_set_here {
                            debug!("variable set here");
                        }
                        if visited_thrice {
                            debug!("do visited thrice");
                            if variable_set_here {
                                debug!("unsetting variable {}", variable);
                                Rc::make_mut(&mut self.variables)[variable] = None;
                            }
                            if self.go_up() {
                                debug!("done");
                                return None;
                            }
                        } else if visited_twice {
                            if variable_set_here {
                                debug!("visiting high child");
                                debug!("setting variable {} to true", variable);
                                Rc::make_mut(&mut self.variables)[variable] = Some(true);
                                self.go_down(high);
                            }
                        } else if visited_once {
                            debug!("do visited once");
                            debug!("going down following value {}", value);
                            self.go_down(if value { high } else { low })
                        } else {
                            debug!("not visited before, output path");
                            return Some(Path::new(&self.path, &self.variables, self.node, None));
                        }
                    } else {
                        debug!("variable {} has no value", variable);
                        if visited_once {
                            debug!("do visited once");
                            debug!("setting variable {} to false", variable);
                            self.variable_set_depth[variable] = depth;
                            Rc::make_mut(&mut self.variables)[variable] = Some(false);
                            self.go_down(low);
                        } else {
                            debug!("not visited before, output path");
                            return Some(Path::new(&self.path, &self.variables, self.node, None));
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use env_logger;

    #[test]
    fn can_iter_leaf_only() {
        let _ = env_logger::init();
        let mut graph = Graph::new();
        let root = graph.one();
        let diagram = OrderedDiagram::from_root(root, 0);
        let paths: Vec<_> = PathIter::new(&diagram, &graph).collect();
        assert_eq!(1, paths.len());
        assert_eq!([Path {
                        path: Rc::new(Vec::new()),
                        variables: Rc::new(Vec::new()),
                        node: root,
                        value: Some(true),
                    }]
                           .as_ref(),
                   paths.as_slice());
    }

    #[test]
    fn can_iter_branch() {
        let _ = env_logger::init();
        let mut graph = Graph::new();
        let zero = graph.zero();
        let one = graph.one();
        let root = graph.branch(0, zero, one);
        let diagram = OrderedDiagram::from_root(root, 1);
        let paths: Vec<_> = PathIter::new(&diagram, &graph).collect();
        let expected = [Path {
                            path: Rc::new(Vec::new()),
                            variables: Rc::new(vec![None]),
                            node: root,
                            value: None,
                        },
                        Path {
                            path: Rc::new(vec![root]),
                            variables: Rc::new(vec![Some(false)]),
                            node: zero,
                            value: Some(false),
                        },
                        Path {
                            path: Rc::new(vec![root]),
                            variables: Rc::new(vec![Some(true)]),
                            node: one,
                            value: Some(true),
                        }];
        println!("expected = {:#?}", expected);
        println!("received = {:#?}", paths);
        assert_eq!(expected.as_ref(), paths.as_slice());
    }

    #[test]
    fn can_iter_tree() {
        let _ = env_logger::init();
        let mut graph = Graph::new();
        let zero = graph.zero();
        let one = graph.one();
        let branch = graph.branch(0, zero, one);
        let root = graph.branch(0, branch, branch);
        let diagram = OrderedDiagram::from_root(root, 1);
        let paths: Vec<_> = PathIter::new(&diagram, &graph).collect();
        let expected = [Path {
                            path: Rc::new(Vec::new()),
                            variables: Rc::new(vec![None]),
                            node: root,
                            value: None,
                        },
                        Path {
                            path: Rc::new(vec![root]),
                            variables: Rc::new(vec![Some(false)]),
                            node: branch,
                            value: None,
                        },
                        Path {
                            path: Rc::new(vec![root, branch]),
                            variables: Rc::new(vec![Some(false)]),
                            node: zero,
                            value: Some(false),
                        },
                        Path {
                            path: Rc::new(vec![root]),
                            variables: Rc::new(vec![Some(true)]),
                            node: branch,
                            value: None,
                        },
                        Path {
                            path: Rc::new(vec![root, branch]),
                            variables: Rc::new(vec![Some(true)]),
                            node: one,
                            value: Some(true),
                        }];
        println!("expected = {:#?}", expected);
        println!("received = {:#?}", paths);
        assert_eq!(expected.as_ref(), paths.as_slice());
    }

    #[test]
    fn can_iter_two_variable_tree() {
        let _ = env_logger::init();
        let mut graph = Graph::new();
        let zero = graph.zero();
        let one = graph.one();
        let branch = graph.branch(1, zero, one);
        let root = graph.branch(0, branch, branch);
        let diagram = OrderedDiagram::from_root(root, 2);
        let paths: Vec<_> = PathIter::new(&diagram, &graph).collect();
        let expected = [Path {
                            path: Rc::new(Vec::new()),
                            variables: Rc::new(vec![None, None]),
                            node: root,
                            value: None,
                        },
                        Path {
                            path: Rc::new(vec![root]),
                            variables: Rc::new(vec![Some(false), None]),
                            node: branch,
                            value: None,
                        },
                        Path {
                            path: Rc::new(vec![root, branch]),
                            variables: Rc::new(vec![Some(false), Some(false)]),
                            node: zero,
                            value: Some(false),
                        },
                        Path {
                            path: Rc::new(vec![root, branch]),
                            variables: Rc::new(vec![Some(false), Some(true)]),
                            node: one,
                            value: Some(true),
                        },
                        Path {
                            path: Rc::new(vec![root]),
                            variables: Rc::new(vec![Some(true), None]),
                            node: branch,
                            value: None,
                        },
                        Path {
                            path: Rc::new(vec![root, branch]),
                            variables: Rc::new(vec![Some(true), Some(false)]),
                            node: zero,
                            value: Some(false),
                        },
                        Path {
                            path: Rc::new(vec![root, branch]),
                            variables: Rc::new(vec![Some(true), Some(true)]),
                            node: one,
                            value: Some(true),
                        }];
        println!("expected = {:#?}", expected);
        println!("received = {:#?}", paths);
        assert_eq!(expected.as_ref(), paths.as_slice());
    }
}
