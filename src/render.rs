use diagram::{Graph, Node, OrderedDiagram};
use dot;
use std::borrow::Cow;
use std::io;
use walk::PathIter;

type Nd = usize;
type Ed = (usize, bool, usize);

struct DiagramRenderer<'a> {
    diagram: OrderedDiagram,
    graph: &'a Graph,
}


fn edge_label(e: &Ed) -> dot::LabelText<'static> {
    if e.1 {
        dot::LabelText::label("high")
    } else {
        dot::LabelText::label("low")
    }
}

fn node_label(graph: &Graph, n: &Nd) -> dot::LabelText<'static> {
    match graph.expand(*n) {
        Node::Branch {
            variable,
            low: _,
            high: _,
        } => dot::LabelText::label(format!("Branch({})", variable)),
        Node::Leaf { value } => dot::LabelText::label(format!("Leaf({})", value)),
    }
}

fn node_id(n: &Nd) -> dot::Id<'static> {
    dot::Id::new(format!("Node{}", n)).unwrap()
}


impl<'a> dot::Labeller<'a, Nd, Ed> for DiagramRenderer<'a> {
    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new(format!("Diagram{}", self.diagram.root)).unwrap()
    }

    fn node_id(&'a self, n: &Nd) -> dot::Id<'a> {
        node_id(n)
    }

    fn edge_label(&'a self, e: &Ed) -> dot::LabelText<'a> {
        edge_label(e)
    }

    fn node_label(&'a self, n: &Nd) -> dot::LabelText<'a> {
        node_label(self.graph, n)
    }
}

impl<'a> dot::GraphWalk<'a, Nd, Ed> for DiagramRenderer<'a> {
    fn nodes(&'a self) -> dot::Nodes<'a, Nd> {
        let mut nodes: Vec<Nd> = PathIter::new(&self.diagram, self.graph)
            .map(|path| path.node)
            .collect();
        nodes.sort();
        nodes.dedup();
        return Cow::Owned(nodes);
    }

    fn edges(&'a self) -> dot::Edges<'a, Ed> {
        let mut edges = Vec::new();
        for path in PathIter::new(&self.diagram, self.graph) {
            if let Node::Branch {
                       variable: _,
                       low,
                       high,
                   } = self.graph.expand(path.node) {
                edges.push((path.node, false, low));
                edges.push((path.node, true, high));
            }
        }
        return Cow::Owned(edges);
    }

    fn source<'b>(&'a self, e: &'b Ed) -> Nd {
        e.0
    }

    fn target<'b>(&'a self, e: &'b Ed) -> Nd {
        e.2
    }
}

pub fn render_diagram<W>(output: &mut W, diagram: OrderedDiagram, graph: &Graph) -> io::Result<()>
    where W: io::Write
{
    let renderer = DiagramRenderer { diagram, graph };
    dot::render(&renderer, output)
}

struct GraphRenderer<'a> {
    graph: &'a Graph,
}

impl<'a> dot::Labeller<'a, Nd, Ed> for GraphRenderer<'a> {
    fn graph_id(&'a self) -> dot::Id<'a> {
        dot::Id::new("WholeGraph").unwrap()
    }

    fn node_id(&'a self, n: &Nd) -> dot::Id<'a> {
        node_id(n)
    }

    fn edge_label(&'a self, e: &Ed) -> dot::LabelText<'a> {
        edge_label(e)
    }

    fn node_label(&'a self, n: &Nd) -> dot::LabelText<'a> {
        node_label(self.graph, n)
    }
}

impl<'a> dot::GraphWalk<'a, Nd, Ed> for GraphRenderer<'a> {
    fn nodes(&'a self) -> dot::Nodes<'a, Nd> {
        Cow::Owned((0..self.graph.len()).collect::<Vec<_>>())
    }

    fn edges(&'a self) -> dot::Edges<'a, Ed> {
        let mut edges = Vec::new();
        for node in 0..self.graph.len() {
            if let Node::Branch {
                       variable: _,
                       low,
                       high,
                   } = self.graph.expand(node) {
                edges.push((node, false, low));
                edges.push((node, true, high));
            }
        }
        return Cow::Owned(edges);
    }

    fn source<'b>(&'a self, e: &'b Ed) -> Nd {
        e.0
    }

    fn target<'b>(&'a self, e: &'b Ed) -> Nd {
        e.2
    }
}

pub fn render_whole_graph<W>(output: &mut W, graph: &Graph) -> io::Result<()>
    where W: io::Write
{
    let renderer = GraphRenderer { graph };
    dot::render(&renderer, output)
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;

    #[test]
    fn can_render_two_variable_tree() {
        let mut graph = Graph::new();
        let zero = graph.zero();
        let one = graph.one();
        let branch = graph.branch(1, zero, one);
        let root = graph.branch(0, branch, branch);
        let diagram = OrderedDiagram::from_root(root, 2);
        let mut f = File::create("test_output/two_variable_tree.dot").unwrap();
        render_diagram(&mut f, diagram, &graph).unwrap();
    }
}
