use diagram::{Graph, Node, OrderedDiagram};
use rand::Rng;

pub fn walk_diagram<R, F>(rng: &mut R, diagram: &OrderedDiagram, graph: &Graph, mut f: F)
    where F: FnMut(&[usize], &[Option<bool>], usize) -> bool,
          R: Rng
{
    let variable_count = diagram.order.len();
    let tree_height_guess = variable_count;
    let mut path = Vec::with_capacity(tree_height_guess);
    let mut visit_count = Vec::with_capacity(tree_height_guess);
    let mut variables = vec![None; variable_count];
    let mut variable_set_depth = vec![None; variable_count];
    let mut node = diagram.root;
    loop {
        debug!("==========");
        debug!("node: {:?}", node);
        debug!("path: {:?}", path);
        debug!("visit_count: {:?}", visit_count);
        debug!("variables: {:?}", variables);
        debug!("variable_set_depth: {:?}", variable_set_depth);
        match graph.expand(node) {
            Node::Leaf { value: _ } => {
                if f(&path, &variables, node) {
                    return;
                }
                debug!("leaf");
                // Back up a level.
                if let Some(n) = path.pop() {
                    node = n;
                    continue;
                } else {
                    return;
                }
            }
            Node::Branch {
                variable,
                low,
                high,
            } => {
                let depth = path.len();
                if let (Some(mut value), Some(set_depth)) =
                    (variables[variable], variable_set_depth[variable]) {
                    debug!("repeated variable");
                    if visit_count.len() <= depth {
                        debug!("new depth");
                        if f(&path, &variables, node) {
                            return;
                        }
                        // We aren't returning up the tree, but another node decided the variables
                        // value for us.
                        visit_count.push(1);
                        path.push(node);
                    } else if visit_count[depth] == 1 {
                        debug!("already visited");
                        // We are returning up the tree.
                        if depth == set_depth {
                            // We can reassign the variable here.
                            value = !value;
                            variables[variable] = Some(value);
                            variable_set_depth[variable] = Some(depth);
                        }
                        visit_count[depth] = 2;
                        path.push(node);
                    } else {
                        debug!("finished visit");
                        // We are returning up the tree, and have tried both values here.
                        if depth == set_depth {
                            variables[variable] = None;
                            variable_set_depth[variable] = None;
                        }
                        if let Some(n) = path.pop() {
                            visit_count.pop();
                            node = n;
                            continue;
                        } else {
                            return;
                        }
                    }
                    node = if value { high } else { low };
                } else {
                    debug!("new variable");
                    if f(&path, &variables, node) {
                        return;
                    }
                    let value: bool = rng.gen();
                    variables[variable] = Some(value);
                    variable_set_depth[variable] = Some(depth);
                    visit_count.push(1);
                    path.push(node);
                    node = if value { high } else { low };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use env_logger;
    use rand::SeedableRng;
    use rand::XorShiftRng;

    #[test]
    fn can_walk_leaf_only() {
        let _ = env_logger::init();
        let mut rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        let mut graph = Graph::new();
        let root = graph.one();
        let diagram = OrderedDiagram::from_root(root, 0);
        let mut nodes_visited: usize = 0;
        walk_diagram(&mut rng, &diagram, &graph, |path, variables, node| {
            debug!("VISIT");
            assert_eq!(root, node);
            assert_eq!(path, &[]);
            assert_eq!(variables, &[]);
            nodes_visited += 1;
            false
        });
        assert_eq!(1, nodes_visited);
    }

    #[test]
    fn can_walk_branch() {
        let _ = env_logger::init();
        let mut rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        let mut graph = Graph::new();
        let zero = graph.zero();
        let one = graph.one();
        let root = graph.branch(0, zero, one);
        let diagram = OrderedDiagram::from_root(root, 1);
        let mut nodes_visited: usize = 0;
        walk_diagram(&mut rng, &diagram, &graph, |_path, _variables, _node| {
            debug!("VISIT");
            nodes_visited += 1;
            false
        });
        assert_eq!(3, nodes_visited);
    }

    #[test]
    fn can_walk_tree() {
        let _ = env_logger::init();
        let mut rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        let mut graph = Graph::new();
        let zero = graph.zero();
        let one = graph.one();
        let branch = graph.branch(0, zero, one);
        let root = graph.branch(0, branch, branch);
        let diagram = OrderedDiagram::from_root(root, 1);
        let mut nodes_visited: usize = 0;
        walk_diagram(&mut rng, &diagram, &graph, |_path, _variables, _node| {
            debug!("VISIT");
            nodes_visited += 1;
            false
        });
        assert_eq!(7, nodes_visited);
    }
}
