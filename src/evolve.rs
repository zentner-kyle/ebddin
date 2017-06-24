#![allow(unused_variables)]
use bit_vec::BitVec;
use diagram::{Branch, Graph, Node, OrderedDiagram};
use evolution_strategies::{Engine, Problem, Strategy};
use rand::Rng;
use random::choose_from_iter;
use std::cmp::{Ordering, PartialOrd};
use std::collections::HashSet;
use std::collections::hash_map::{Entry, HashMap};
use std::sync::Arc;
use walk::PathIter;

fn generate_branch<R>(rng: &mut R, variable_count: usize, graph: &mut Graph) -> usize
    where R: Rng
{
    let variable = rng.gen_range(0, variable_count);
    let low = rng.gen_range(0, graph.len());
    let high = rng.gen_range(0, graph.len());
    if low == high {
        low
    } else {
        graph.branch(variable, low, high)
    }
}

#[derive(Clone, Debug)]
pub struct DiagramIndividual {
    fitness: f64,
    diagram: OrderedDiagram,
}

pub struct DiagramProblem<F>
    where F: Fn(&Graph, &OrderedDiagram) -> f64
{
    variable_count: usize,
    graph: Graph,
    fitness: Arc<F>,
}

impl<F> Clone for DiagramProblem<F>
    where F: Fn(&Graph, &OrderedDiagram) -> f64
{
    fn clone(&self) -> Self {
        DiagramProblem {
            variable_count: self.variable_count,
            graph: self.graph.clone(),
            fitness: self.fitness.clone(),
        }
    }
}

impl<F> Problem for DiagramProblem<F>
    where F: Fn(&Graph, &OrderedDiagram) -> f64
{
    type Individual = DiagramIndividual;
    fn initialize<R>(&mut self, count: usize, rng: &mut R) -> Vec<Self::Individual>
        where R: Rng
    {

        (0..count)
            .map(|_| {
                let mut order: Vec<_> = (0..self.variable_count).collect();
                rng.shuffle(&mut order);
                let root = generate_branch(rng, self.variable_count, &mut self.graph);
                let diagram = OrderedDiagram {
                    root,
                    order: Arc::new(order),
                };
                let fitness = (self.fitness)(&self.graph, &diagram);
                DiagramIndividual { diagram, fitness }
            })
            .collect()
    }

    fn mutate<R>(&mut self, child: &mut Self::Individual, rng: &mut R) -> bool
        where R: Rng
    {
        let graph = &mut self.graph;
        match rng.gen_range(0, 6) {
            0 => {
                mutate_n1(rng, &mut child.diagram, graph);
                false
            }
            1 => {
                mutate_n1_inv(rng, &mut child.diagram, graph);
                false
            }
            2 => {
                mutate_n2(rng, &mut child.diagram, graph);
                false
            }
            3 => {
                mutate_n2_inv(rng, &mut child.diagram, graph);
                false
            }
            4 => {
                mutate_n3(rng, &mut child.diagram, graph);
                false
            }
            5 => {
                if mutate_a1(rng, &mut child.diagram, graph) {
                    child.fitness = (self.fitness)(graph, &child.diagram);
                    true
                } else {
                    false
                }
            }
            _ => unreachable!(),
        }
    }

    fn compare<R>(&mut self,
                  a: &Self::Individual,
                  b: &Self::Individual,
                  _rng: &mut R)
                  -> Option<Ordering>
        where R: Rng
    {
        a.fitness.partial_cmp(&b.fitness)
    }
}

pub fn evolve_diagrams<F, R>(rng: R,
                             strategy: Strategy,
                             variable_count: usize,
                             generations: usize,
                             fitness: F)
                             -> Engine<DiagramProblem<F>, R>
    where F: Fn(&Graph, &OrderedDiagram) -> f64,
          R: Rng
{
    let graph = Graph::new();
    let problem = DiagramProblem {
        variable_count,
        graph,
        fitness: Arc::new(fitness),
    };
    let mut engine = Engine::new(problem, strategy, rng);
    for i in 0..generations {
        engine.run_generation();
        if i % 100 == 0 && i > 0 {
            println!("Completed generation {}", i);
        }
    }
    engine
}

fn random_bitvec<R>(rng: &mut R, size: usize) -> BitVec
    where R: Rng
{
    let mut result = BitVec::with_capacity(size);
    for _ in 0..size {
        result.push(rng.gen());
    }
    result
}

fn make_parent_graph(graph: &Graph, diagram: &OrderedDiagram) -> HashMap<usize, Vec<usize>> {
    let mut parent_graph = HashMap::new();
    for visit in PathIter::new(diagram, graph) {
        if let Some(parent) = visit.path.last() {
            parent_graph
                .entry(visit.node)
                .or_insert_with(|| Vec::new())
                .push(*parent);
        }
    }
    return parent_graph;
}

fn make_ancestor_set<I>(graph: &Graph, diagram: &OrderedDiagram, nodes: I) -> HashSet<usize>
    where I: Iterator<Item = usize>
{
    let roots: HashSet<usize> = nodes.collect();
    let mut ancestors = HashSet::new();
    for visit in PathIter::new(diagram, graph) {
        if roots.contains(&visit.node) {
            for node in visit.path.as_slice() {
                ancestors.insert(*node);
            }
        }
    }
    return ancestors;
}

fn rebuild_diagram(graph: &mut Graph,
                   diagram: &OrderedDiagram,
                   original: usize,
                   replacement: usize)
                   -> usize {
    let mut replacements: HashMap<usize, usize> = HashMap::new();
    replacements.insert(original, replacement);
    rebuild_diagram_from_replacements(graph, diagram, replacements)
}

fn add_parents_if_ready(node: usize,
                        graph: &Graph,
                        ready: &mut Vec<usize>,
                        parent_graph: &HashMap<usize, Vec<usize>>,
                        replacements: &HashMap<usize, usize>,
                        ancestor_set: &HashSet<usize>) {
    for &parent in parent_graph
            .get(&node)
            .map(|v| v.as_slice())
            .unwrap_or(&[]) {
        let Branch {
            variable,
            low,
            high,
        } = graph.expand_branch(parent);
        if (!ancestor_set.contains(&low) || replacements.contains_key(&low)) &&
           (!ancestor_set.contains(&high) || replacements.contains_key(&high)) {
            ready.push(parent);
        }
    }
}

fn rebuild_diagram_from_replacements(graph: &mut Graph,
                                     diagram: &OrderedDiagram,
                                     mut replacements: HashMap<usize, usize>)
                                     -> usize {
    let parent_graph = make_parent_graph(graph, diagram);
    let ancestor_set = make_ancestor_set(graph, diagram, replacements.keys().cloned());
    let mut ready = Vec::new();
    for &replaced in replacements.keys() {
        add_parents_if_ready(replaced,
                             &graph,
                             &mut ready,
                             &parent_graph,
                             &replacements,
                             &ancestor_set);
    }
    while let Some(node) = ready.pop() {
        let Branch {
            variable,
            low,
            high,
        } = graph.expand_branch(node);
        let low_replacement = replacements.get(&low).cloned().unwrap_or(low);
        let high_replacement = replacements.get(&high).cloned().unwrap_or(high);
        let replacement = graph.branch(variable, low_replacement, high_replacement);
        replacements.insert(node, replacement);
        add_parents_if_ready(node,
                             &graph,
                             &mut ready,
                             &parent_graph,
                             &replacements,
                             &ancestor_set);
    }
    return *replacements.get(&diagram.root).unwrap_or(&diagram.root);
}

fn mutate_n1<R>(rng: &mut R, diagram: &mut OrderedDiagram, graph: &mut Graph) -> bool
    where R: Rng
{
    // Remove a random redundant test.
    if let Some((to_fix, replacement)) =
        choose_from_iter(rng,
                         PathIter::new(diagram, graph).filter_map(|path| {
            match graph.expand(path.node) {
                Node::Branch {
                    variable: _,
                    low,
                    high,
                } if low == high => Some((path, low)),
                _ => None,
            }
        })) {
        diagram.root = rebuild_diagram(graph, diagram, to_fix.node, replacement);
        return true;
    } else {
        return false;
    }
}

fn mutate_n1_inv<R>(rng: &mut R, diagram: &mut OrderedDiagram, graph: &mut Graph) -> bool
    where R: Rng
{
    // Insert a random redundant test.
    let parent_graph = make_parent_graph(graph, diagram);
    if let Some((to_fix, min_variable_idx, last_allowed_variable_idx)) =
        choose_from_iter(rng,
                         PathIter::new(diagram, graph).filter_map(|visit| {
            let next_variable_idx;
            if let Node::Branch { variable, .. } = graph.expand(visit.node) {
                let next_variable = variable;
                next_variable_idx = diagram
                    .order
                    .iter()
                    .position(|&v| v == next_variable)
                    .expect("All variables should be in the order");
            } else {
                next_variable_idx = diagram.order.len();
            }
            let mut max_prev_variable_idx = -1;
            if let Some(parents) = parent_graph.get(&visit.node) {
                for &parent in parents {
                    let variable = graph.expand_branch(parent).variable;
                    let variable_index = diagram
                        .order
                        .iter()
                        .position(|&v| v == variable)
                        .expect("All variables should be in the order");
                    if variable_index as isize > max_prev_variable_idx {
                        max_prev_variable_idx = variable_index as isize;
                    }
                }
            }
            let min_variable_idx = (max_prev_variable_idx + 1) as usize;
            if next_variable_idx > min_variable_idx {
                return Some((visit, min_variable_idx, next_variable_idx - 1));
            }
            return None;
        })) {
        let variable_index = rng.gen_range(min_variable_idx, 1 + last_allowed_variable_idx);
        let fixed = graph.branch(diagram.order[variable_index], to_fix.node, to_fix.node);
        // If we didn't replace the root.
        if let Some(&parent) = to_fix.path.last() {
            let Branch {
                variable,
                low,
                high,
            } = graph.expand_branch(parent);
            let replacement;
            if to_fix.node == low {
                replacement = graph.branch(variable, fixed, high);
            } else {
                replacement = graph.branch(variable, low, fixed);
            }
            diagram.root = rebuild_diagram(graph, diagram, parent, replacement);
        } else {
            diagram.root = fixed;
        }
        return true;
    } else {
        return false;
    }
}

fn mutate_n2<R>(rng: &mut R, diagram: &mut OrderedDiagram, graph: &mut Graph) -> bool
    where R: Rng
{
    // Remove a redundant non-terminal.
    if let Some((to_fix, variable, child)) =
        choose_from_iter(rng,
                         PathIter::new(diagram, graph).filter_map(|path| {
            if let Node::Branch {
                       variable,
                       low,
                       high,
                   } = graph.expand(path.node) {
                if let (Node::Branch {
                            low: low_low,
                            high: low_high,
                            ..
                        },
                        Node::Branch {
                            low: high_low,
                            high: high_high,
                            ..
                        }) = (graph.expand(low), graph.expand(low)) {
                    if low_low == high_low && low_high == high_high {
                        return Some((path, variable, low));
                    }
                }
            }
            return None;
        })) {
        let replacement = graph.branch(variable, child, child);
        diagram.root = rebuild_diagram(graph, diagram, to_fix.node, replacement);
        return true;
    } else {
        return false;
    }
}

fn mutate_n2_inv<R>(rng: &mut R, diagram: &mut OrderedDiagram, graph: &mut Graph) -> bool
    where R: Rng
{
    // Insert a redundant non-terminal.
    if let Some((to_fix, variable, (child, child_variable, low, high))) =
        choose_from_iter(rng,
                         PathIter::new(diagram, graph).filter_map(|path| {
            if let Node::Branch {
                       low,
                       high,
                       variable,
                   } = graph.expand(path.node) {
                if low == high {
                    if let Node::Branch {
                               variable: child_variable,
                               low: child_low,
                               high: child_high,
                           } = graph.expand(low) {
                        return Some((path, variable, (low, child_variable, child_low, child_high)));
                    }
                }
            }
            return None;
        })) {
        let child_dup = graph.branch(child_variable, low, high);
        let replacement = graph.branch(variable, child, child_dup);
        diagram.root = rebuild_diagram(graph, diagram, to_fix.node, replacement);
        return true;
    } else {
        return false;
    }
}

fn mutate_n3<R>(rng: &mut R, diagram: &mut OrderedDiagram, graph: &mut Graph) -> bool
    where R: Rng
{
    // Swap adjacent variables in order.
    if diagram.order.len() < 2 {
        return false;
    }
    let first_variable_index = rng.gen_range(0, diagram.order.len() - 1);
    let second_variable_index = first_variable_index + 1;
    let first_variable = diagram.order[first_variable_index];
    let second_variable = diagram.order[second_variable_index];
    let mut to_replace = Vec::new();
    for visit in PathIter::new(diagram, graph) {
        if let Node::Branch {
                   low,
                   high,
                   variable,
               } = graph.expand(visit.node) {
            let low_low;
            let low_high;
            let high_low;
            let high_high;
            match (graph.expand(low), graph.expand(high)) {
                (Node::Branch {
                     low: ll,
                     high: lh,
                     variable: lv,
                 },
                 Node::Branch {
                     low: hl,
                     high: hh,
                     variable: hv,
                 }) if lv == second_variable && hv == second_variable => {
                    low_low = ll;
                    low_high = lh;
                    high_low = hl;
                    high_high = hh;
                }
                (Node::Branch {
                     low: ll,
                     high: lh,
                     variable: lv,
                 },
                 _) if lv == second_variable => {
                    low_low = ll;
                    low_high = lh;
                    high_low = high;
                    high_high = high;
                }
                (_,
                 Node::Branch {
                     low: hl,
                     high: hh,
                     variable: hv,
                 }) if hv == second_variable => {
                    low_low = low;
                    low_high = low;
                    high_low = hl;
                    high_high = hh;
                }
                (_, _) => {
                    low_low = low;
                    low_high = low;
                    high_low = high;
                    high_high = high;
                }
            }
            to_replace.push((visit.node, low_low, low_high, high_low, high_high));
        }
    }
    let mut replacements: HashMap<usize, usize> = HashMap::new();
    for (node, ll, lh, hl, hh) in to_replace {
        if let Entry::Vacant(entry) = replacements.entry(node) {
            let new_low = graph.branch(first_variable, ll, hl);
            let new_high = graph.branch(first_variable, lh, hh);
            let replacement = graph.branch(second_variable, new_low, new_high);
            entry.insert(replacement);
        }
    }
    Arc::make_mut(&mut diagram.order)[first_variable_index] = second_variable;
    Arc::make_mut(&mut diagram.order)[second_variable_index] = first_variable;
    diagram.root = rebuild_diagram_from_replacements(graph, diagram, replacements);
    return true;
}

fn mutate_a1<R>(rng: &mut R, diagram: &mut OrderedDiagram, graph: &mut Graph) -> bool
    where R: Rng
{
    // Change a random non-terminal to point to a new vertex with a later variable.
    if let Some((to_fix, target_variable, mut low, mut high)) =
        choose_from_iter(rng,
                         PathIter::new(diagram, graph).filter_map(|path| {
            if let Node::Branch {
                       low,
                       high,
                       variable,
                   } = graph.expand(path.node) {
                return Some((path, variable, low, high));
            }
            return None;
        })) {
        let target_variable_index = diagram
            .order
            .iter()
            .position(|&v| v == target_variable)
            .expect("All variables should be in the order");
        if let Some(target) = choose_from_iter(rng,
                                               PathIter::new(diagram, graph)
                                                   .filter_map(|visit| {
            match graph.expand(visit.node) {
                Node::Branch { variable, .. } => {
                    let variable_index = diagram
                        .order
                        .iter()
                        .position(|&v| v == variable)
                        .expect("All variables should be in the order");
                    if variable_index > target_variable_index {
                        Some(visit.node)
                    } else {
                        None
                    }
                }
                Node::Leaf { .. } => None,
            }
        })
                                                   .chain([0, 1].iter().cloned())) {
            if rng.gen() {
                // Change low.
                low = target;
            } else {
                // Change high.
                high = target;
            }
            let replacement = graph.branch(target_variable, low, high);
            diagram.root = rebuild_diagram(graph, diagram, to_fix.node, replacement);
            return true;
        }
    }
    return false;
}


#[cfg(test)]
mod tests {
    use super::*;
    use evaluate_diagram;
    use rand::SeedableRng;
    use rand::XorShiftRng;
    use render::{render_diagram, render_whole_graph};
    use std::fs::File;

    #[test]
    fn evolve_can_evolve_1() {
        let test_name = "evolve_can_evolve_1";
        let rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        let variable_count = 1;
        let strategy = Strategy::MuLambda { mu: 2, lambda: 5 };
        let zero_bitvec = BitVec::from_elem(variable_count, false);
        let fitness = |graph: &_, diagram: &OrderedDiagram| if evaluate_diagram(graph,
                                                                                diagram.root,
                                                                                &zero_bitvec) {
            return 1.0f64;
        } else {
            return 0.0f64;
        };
        let engine = evolve_diagrams(rng, strategy, variable_count, 10, fitness);
        let graph = &engine.problem().graph;
        {
            let mut f = File::create(format!("test_output/{}_graph.dot", test_name)).unwrap();
            render_whole_graph(&mut f, &graph).unwrap();
        }
        for (idx, diagram) in engine.population().map(|i| &i.diagram).enumerate() {
            let mut f = File::create(format!("test_output/diagram_{}_{}.dot", test_name, idx))
                .unwrap();
            render_diagram(&mut f, diagram.clone(), graph).unwrap();
            assert!(evaluate_diagram(graph, diagram.root, &zero_bitvec));
        }
    }

    #[test]
    fn evolve_can_evolve_identity() {
        let test_name = "evolve_can_evolve_identity";
        let rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        let variable_count = 1;
        let strategy = Strategy::MuLambda { mu: 5, lambda: 10 };

        let zero_bitvec = BitVec::from_elem(variable_count, false);
        let one_bitvec = BitVec::from_elem(variable_count, true);
        let fitness = |graph: &_, diagram: &OrderedDiagram| {
            let mut f = 0.0;
            if evaluate_diagram(graph, diagram.root, &zero_bitvec) == false {
                f += 1.0;
            }
            if evaluate_diagram(graph, diagram.root, &one_bitvec) == true {
                f += 1.0;
            }
            return f;
        };
        let engine = evolve_diagrams(rng, strategy, variable_count, 10, fitness);
        let graph = &engine.problem().graph;
        {
            let mut f = File::create(format!("test_output/{}_graph.dot", test_name)).unwrap();
            render_whole_graph(&mut f, &graph).unwrap();
        }
        for (idx, diagram) in engine.population().map(|i| &i.diagram).enumerate() {
            let mut f = File::create(format!("test_output/diagram_{}_{}.dot", test_name, idx))
                .unwrap();
            render_diagram(&mut f, diagram.clone(), graph).unwrap();
            assert!((engine.problem().fitness)(graph, diagram) == 2.0);
        }
    }

    #[test]
    fn evolve_can_evolve_two_bit_parity() {
        let test_name = "evolve_can_evolve_two_bit_parity";
        let rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        let variable_count = 2;
        let strategy = Strategy::MuLambda { mu: 5, lambda: 10 };
        let zero_bitvec: BitVec = [false, false].iter().cloned().collect();
        let one_bitvec: BitVec = [true, false].iter().cloned().collect();
        let two_bitvec: BitVec = [false, true].iter().cloned().collect();
        let three_bitvec: BitVec = [true, true].iter().cloned().collect();
        let fitness = |graph: &Graph, diagram: &OrderedDiagram| {
            let mut f = 0.0;
            if evaluate_diagram(graph, diagram.root, &zero_bitvec) == false {
                f += 1.0;
            }
            if evaluate_diagram(graph, diagram.root, &one_bitvec) == true {
                f += 1.0;
            }
            if evaluate_diagram(graph, diagram.root, &two_bitvec) == true {
                f += 1.0;
            }
            if evaluate_diagram(graph, diagram.root, &three_bitvec) == false {
                f += 1.0;
            }
            if f > 3.0 {
                println!("f = {}", f);
            }
            return f;
        };
        let engine = evolve_diagrams(rng, strategy, variable_count, 10, fitness);
        let graph = &engine.problem().graph;
        {
            let mut f = File::create(format!("test_output/{}_graph.dot", test_name)).unwrap();
            render_whole_graph(&mut f, &graph).unwrap();
        }
        for (idx, diagram) in engine.population().map(|i| &i.diagram).enumerate() {
            let mut f = File::create(format!("test_output/diagram_{}_{}.dot", test_name, idx))
                .unwrap();
            render_diagram(&mut f, diagram.clone(), graph).unwrap();
            //assert!((engine.problem().fitness)(graph, diagram) == 4.0);
        }
    }

    #[test]
    fn can_mutate_n1_inv_above_leaf() {
        let mut rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        let mut graph = Graph::new();
        let branch = graph.branch(0, 0, 1);
        let mut diagram = OrderedDiagram::from_root(branch, 2);
        assert!(mutate_n1_inv(&mut rng, &mut diagram, &mut graph));
        assert!(mutate_n1_inv(&mut rng, &mut diagram, &mut graph));
        assert!(!mutate_n1_inv(&mut rng, &mut diagram, &mut graph));
        let mut f = File::create("test_output/can_mutate_n1_inv_above_leaf.dot").unwrap();
        render_diagram(&mut f, diagram.clone(), &graph).unwrap();
    }

    #[test]
    fn can_mutate_n1_inv_at_leaf() {
        let mut rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        let mut graph = Graph::new();
        let mut diagram = OrderedDiagram::from_root(0, 1);
        assert!(mutate_n1_inv(&mut rng, &mut diagram, &mut graph));
        assert!(!mutate_n1_inv(&mut rng, &mut diagram, &mut graph));
        let mut f = File::create("test_output/can_mutate_n1_inv_at_leaf.dot").unwrap();
        render_diagram(&mut f, diagram.clone(), &graph).unwrap();
    }
}
