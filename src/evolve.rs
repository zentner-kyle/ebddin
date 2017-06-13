#![allow(unused_variables)]
use bit_vec::BitVec;
use diagram::{Graph, Node, OrderedDiagram};
use rand::Rng;
use random::choose_from_iter;
use std::collections::HashSet;
use std::collections::hash_map::HashMap;
use std::sync::Arc;
use walk::PathIter;

#[derive(Clone, Copy, Debug)]
pub enum EvolutionStrategy {
    Pairwise {
        population: usize,
        generations: usize,
    },
}

fn population(es: EvolutionStrategy) -> usize {
    match es {
        EvolutionStrategy::Pairwise {
            generations,
            population,
        } => population,
    }
}

fn generations(es: EvolutionStrategy) -> usize {
    match es {
        EvolutionStrategy::Pairwise {
            generations,
            population,
        } => generations,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RDDParams {
    variable_count: usize,
    mutation_count: usize,
}

fn generate_branch<R>(rng: &mut R, params: RDDParams, graph: &mut Graph) -> usize
    where R: Rng
{
    let variable = rng.gen_range(0, params.variable_count);
    let low = rng.gen_range(0, graph.len());
    let high = rng.gen_range(0, graph.len());
    if low == high {
        low
    } else {
        graph.branch(variable, low, high)
    }
}

struct StrategySelector {
    strategy: EvolutionStrategy,
    pending_children: Vec<(OrderedDiagram, f64)>,
    pending_parents: Option<(usize, usize)>,
}

impl StrategySelector {
    fn new(es: EvolutionStrategy) -> Self {
        StrategySelector {
            strategy: es,
            pending_children: Vec::new(),
            pending_parents: None,
        }
    }

    fn choose_parents<R>(&mut self, rng: &mut R, diagrams: &[OrderedDiagram]) -> (usize, usize)
        where R: Rng
    {
        let a_idx = rng.gen_range(0, diagrams.len());
        let mut b_idx = rng.gen_range(0, diagrams.len());
        while b_idx == a_idx {
            b_idx = rng.gen_range(0, diagrams.len());
        }
        let result = (a_idx, b_idx);
        self.pending_parents = Some(result);
        self.pending_children.clear();
        result
    }

    fn add_child(&mut self, child: OrderedDiagram, fitness: f64) {
        self.pending_children.push((child, fitness));
    }

    fn output_generation(&mut self,
                         individuals: &mut Vec<OrderedDiagram>,
                         fitness: &mut Vec<f64>) {
        let parents = self.pending_parents
            .take()
            .expect("Need to be in a generation to output it.");
        let mut children_iter = self.pending_children.drain(..);
        match (children_iter.next(), children_iter.next()) {
            (Some((child_0, f_0)), Some((child_1, f_1))) => {
                if parents.0 != parents.1 {
                    individuals[parents.0] = child_0;
                    fitness[parents.0] = f_0;
                    individuals[parents.1] = child_1;
                    fitness[parents.1] = f_1;
                } else {
                    individuals[parents.0] = child_0;
                    fitness[parents.0] = f_0;
                    individuals.push(child_1);
                    fitness.push(f_1);
                }
                for (child, f) in children_iter {
                    individuals.push(child);
                    fitness.push(f);
                }
            }
            (Some((child_0, f_0)), None) => {
                if parents.0 != parents.1 {
                    individuals[parents.0] = child_0;
                    fitness[parents.0] = f_0;
                    individuals.remove(parents.1);
                    fitness.remove(parents.1);
                } else {
                    individuals[parents.0] = child_0;
                    fitness[parents.0] = f_0;
                }
            }
            (None, None) => {
                if parents.0 != parents.1 {
                    individuals.remove(parents.0);
                    fitness.remove(parents.0);
                    individuals.remove(parents.1);
                    fitness.remove(parents.1);
                } else {
                    individuals.remove(parents.0);
                    fitness.remove(parents.0);
                }
            }
            (None, Some(_)) => unreachable!(),
        }
    }
}

pub struct DiagramPopulation {
    graph: Graph,
    diagrams: Vec<OrderedDiagram>,
}

pub fn evolve_diagrams<'a, F, R>(rng: &'a mut R,
                                 es: EvolutionStrategy,
                                 params: RDDParams,
                                 mut fitness: F)
                                 -> DiagramPopulation
    where F: FnMut(&Graph, &OrderedDiagram) -> f64,
          R: Rng
{
    let pop = population(es);
    let mut graph = Graph::with_capacity(pop);
    let mut diagrams: Vec<_> = (0..pop)
        .map(|_| {
            let mut order: Vec<_> = (0..pop).collect();
            rng.shuffle(&mut order);
            let root = generate_branch(rng, params, &mut graph);
            OrderedDiagram {
                root,
                order: Arc::new(order),
            }
        })
        .collect();
    let mut fit = {
        diagrams
            .iter()
            .map(|d| fitness(&graph, d))
            .collect::<Vec<f64>>()
    };
    let mut strategy = StrategySelector::new(es);
    for _ in 0..generations(es) {
        let (a, b) = strategy.choose_parents(rng, &diagrams);
        let parent;
        let fitness_parent;
        let fitness_a = fit[a];
        let fitness_b = fit[b];
        if fitness_a >= fitness_b {
            parent = diagrams[a].clone();
            fitness_parent = fitness_a;
        } else {
            parent = diagrams[b].clone();
            fitness_parent = fitness_b;
        }
        let num_children = 2;
        for _ in 0..num_children {
            let mut child = parent.clone();
            let mut adapted = false;
            for _ in 0..params.mutation_count {
                match rng.gen_range(0, 6) {
                    0 => {
                        mutate_n1(rng, &mut child, &mut graph);
                    }
                    1 => {
                        mutate_n1_inv(rng, &mut child, &mut graph);
                    }
                    2 => {
                        mutate_n2(rng, &mut child, &mut graph);
                    }
                    3 => {
                        mutate_n2_inv(rng, &mut child, &mut graph);
                    }
                    4 => {
                        mutate_n3(rng, &mut child, &mut graph);
                    }
                    5 => {
                        adapted |= mutate_a1(rng, &mut child, &mut graph);
                    }
                    _ => unreachable!(),
                }
            }
            let fitness_child = if adapted {
                fitness(&graph, &child)
            } else {
                fitness_parent
            };
            strategy.add_child(child, fitness_child);
        }
        strategy.output_generation(&mut diagrams, &mut fit);
    }

    return DiagramPopulation { graph, diagrams };
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

fn make_ancestor_set(graph: &Graph, diagram: &OrderedDiagram, node: usize) -> HashSet<usize> {
    let mut ancestors = HashSet::new();
    for visit in PathIter::new(diagram, graph) {
        if visit.node == node {
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
    let ancestor_set = make_ancestor_set(graph, diagram, original);
    let parent_graph = make_parent_graph(graph, diagram);
    let mut replacements: HashMap<usize, usize> = HashMap::new();
    replacements.insert(original, replacement);
    let mut ready = Vec::new();
    ready.push(original);
    while let Some(node) = ready.pop() {
        if let Node::Branch {
                   variable,
                   low,
                   high,
               } = graph.expand(node) {
            let low_replacement = replacements.get(&low).cloned().unwrap_or(low);
            let high_replacement = replacements.get(&high).cloned().unwrap_or(high);
            let replacement = graph.branch(variable, low_replacement, high_replacement);
            replacements.insert(node, replacement);
            for &parent in parent_graph
                    .get(&node)
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]) {
                if let Node::Branch {
                           variable,
                           low,
                           high,
                       } = graph.expand(parent) {
                    if (!ancestor_set.contains(&low) || replacements.contains_key(&low)) &&
                       (!ancestor_set.contains(&high) || replacements.contains_key(&high)) {
                        ready.push(parent);
                    }
                } else {
                    unreachable!();
                }
            }
        } else {
            panic!("Can only replace branch nodes");
        }
    }
    return *replacements
                .get(&diagram.root)
                .expect("Should have reached root");
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
    if let Some((to_fix, min_variable_idx, next_variable_idx)) =
        choose_from_iter(rng,
                         PathIter::new(diagram, graph).filter_map(|visit| {
            let next_variable;
            if let Node::Branch { variable, .. } = graph.expand(visit.node) {
                next_variable = variable;
            } else {
                next_variable = *diagram.order.last().unwrap();
            }
            let next_variable_idx = diagram
                .order
                .iter()
                .position(|&v| v == next_variable)
                .expect("All variables should be in the order");
            let mut max_prev_variable_idx = -1;
            if let Some(parents) = parent_graph.get(&visit.node) {
                for &parent in parents {
                    if let Node::Branch { variable, .. } = graph.expand(parent) {
                        let variable_index = diagram
                            .order
                            .iter()
                            .position(|&v| v == variable)
                            .expect("All variables should be in the order");
                        if variable_index as isize > max_prev_variable_idx {
                            max_prev_variable_idx = variable_index as isize;
                        }
                    } else {
                        unreachable!();
                    }
                }
            }
            let min_variable_idx = (max_prev_variable_idx + 1) as usize;
            if next_variable_idx > min_variable_idx {
                return Some((visit, min_variable_idx, next_variable_idx));
            }
            return None;
        })) {
        let variable_index = rng.gen_range(min_variable_idx, next_variable_idx);
        let fixed = graph.branch(diagram.order[variable_index], to_fix.node, to_fix.node);
        // If we didn't replace the root.
        if let Some(&parent) = to_fix.path.last() {
            if let Node::Branch {
                       variable,
                       low,
                       high,
                   } = graph.expand(parent) {
                let replacement;
                if to_fix.node == low {
                    replacement = graph.branch(variable, fixed, high);
                } else {
                    replacement = graph.branch(variable, low, fixed);
                }
                diagram.root = rebuild_diagram(graph, diagram, parent, replacement);
            } else {
                unreachable!();
            }
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
    return false;
}

fn mutate_a1<R>(rng: &mut R, diagram: &mut OrderedDiagram, graph: &mut Graph) -> bool
    where R: Rng
{
    // Change a random non-terminal to point to a new vertex that is not its own ancestor.
    let mut allowed_nodes = HashSet::new();
    if let Some((to_fix, variable, mut low, mut high)) =
        choose_from_iter(rng,
                         PathIter::new(diagram, graph).filter_map(|path| {
            allowed_nodes.insert(path.node);
            if let Node::Branch {
                       low,
                       high,
                       variable,
                   } = graph.expand(path.node) {
                return Some((path, variable, low, high));
            }
            return None;
        })) {
        for visit in PathIter::new(diagram, graph) {
            if visit.node == to_fix.node {
                for node in visit.path.as_slice() {
                    allowed_nodes.remove(node);
                }
            }
        }
        let allowed_nodes: Vec<usize> = allowed_nodes.into_iter().collect();
        if rng.gen() {
            // Change low.
            low = rng.choose(&allowed_nodes)
                .cloned()
                .expect("branches must have some non-ancestor nodes in their diagram");
        } else {
            // Change high.
            high = rng.choose(&allowed_nodes)
                .cloned()
                .expect("branches must have some non-ancestor nodes in their diagram");
        }
        let replacement = graph.branch(variable, low, high);
        diagram.root = rebuild_diagram(graph, diagram, to_fix.node, replacement);
        return true;
    } else {
        return false;
    }
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
    fn evolve_can_evaluate_1() {
        let mut rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        let pop = 2;
        let strategy = EvolutionStrategy::Pairwise {
            population: pop,
            generations: 10 * pop,
        };
        let params = RDDParams {
            variable_count: 10,
            mutation_count: 10,
        };
        let zero_bitvec = BitVec::from_elem(params.variable_count, false);
        let fitness = |graph: &_, diagram: &OrderedDiagram| if evaluate_diagram(graph,
                                                                                diagram.root,
                                                                                &zero_bitvec) {
            return 1.0f64;
        } else {
            return 0.0f64;
        };
        let evolved_population = evolve_diagrams(&mut rng, strategy, params, fitness);
        {
            let mut f = File::create("test_output/evolve_can_evaluate_1_graph.dot").unwrap();
            render_whole_graph(&mut f, &evolved_population.graph).unwrap();
        }
        for (idx, diagram) in evolved_population.diagrams.iter().enumerate() {
            let mut f = File::create(format!("test_output/diagram_to_compute_1_number{}.dot", idx))
                .unwrap();
            render_diagram(&mut f, diagram.clone(), &evolved_population.graph).unwrap();
            assert!(evaluate_diagram(&evolved_population.graph, diagram.root, &zero_bitvec));
        }
    }
}
