#![allow(unused_variables)]
use bit_vec::BitVec;
use diagram::{Graph, Node, OrderedDiagram};
use rand::Rng;
use std::sync::Arc;

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
                                 fitness: F)
                                 -> DiagramPopulation
    where F: Fn(&Graph, &OrderedDiagram) -> f64,
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

fn mutate_n1<R>(rng: &mut R, diagram: &mut OrderedDiagram, graph: &mut Graph) -> bool
    where R: Rng
{
    // Remove a random redundant test.
    return false;
}

fn mutate_n1_inv<R>(rng: &mut R, diagram: &mut OrderedDiagram, graph: &mut Graph) -> bool
    where R: Rng
{
    // Insert a random redundant test.
    let mut path = Vec::with_capacity(diagram.order.len());
    let mut node = diagram.root;
    loop {
        path.push(node);
        match graph.expand(node) {
            Node::Leaf { value: _ } => {
                break;
            }
            Node::Branch {
                variable,
                low,
                high,
            } => {
                let direction = rng.gen();
                if direction {
                    node = high;
                } else {
                    node = low;
                }
            }
        }
    }
    let step_to_mutate = rng.gen_range(0, path.len());
    // Insert a redundant check *before* step_to_mutate.
    let original = path[step_to_mutate];
    // Figure out what variable to use for the redundant test.
    let variable = if step_to_mutate == 0 {
        // We chose to insert before the first step, so we can use the first variable.
        diagram.order[0]
    } else {
        // Go back one step and get the variable.
        let previous_step = path[step_to_mutate - 1];
        if let Node::Branch {
                   variable: v,
                   low: _,
                   high: _,
               } = graph.expand(previous_step) {
            v
        } else {
            // Wasn't the last step, so it must be a branch.
            unreachable!();
        }
    };
    // Insert the redundant test.
    let mut replacement = graph.branch(variable, original, original);
    for i in (0..step_to_mutate).rev() {
        let original = path[i + 1];
        if let Node::Branch {
                   variable,
                   low,
                   high,
               } = graph.expand(path[i]) {
            if low == original {
                replacement = graph.branch(variable, replacement, high);
            } else {
                replacement = graph.branch(variable, low, replacement);
            }
        } else {
            unreachable!();
        }
    }
    diagram.root = replacement;
    return true;
}

fn mutate_n2<R>(rng: &mut R, diagram: &mut OrderedDiagram, graph: &mut Graph) -> bool
    where R: Rng
{
    // Remove a random redundant test.
    return false;
}

fn mutate_n2_inv<R>(rng: &mut R, diagram: &mut OrderedDiagram, graph: &mut Graph) -> bool
    where R: Rng
{
    // Remove a random redundant test.
    return false;
}

fn mutate_n3<R>(rng: &mut R, diagram: &mut OrderedDiagram, graph: &mut Graph) -> bool
    where R: Rng
{
    // Remove a random redundant test.
    return false;
}

fn mutate_a1<R>(rng: &mut R, diagram: &mut OrderedDiagram, graph: &mut Graph) -> bool
    where R: Rng
{
    // Remove a random redundant test.
    return false;
}


#[cfg(test)]
mod tests {
    use super::*;
    use evaluate_diagram;
    use rand::SeedableRng;
    use rand::XorShiftRng;

    #[test]
    fn evolve_can_evaluate_1() {
        let mut rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        let pop = 10;
        let strategy = EvolutionStrategy::Pairwise {
            population: pop,
            generations: 100 * pop,
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
        for diagram in &evolved_population.diagrams {
            assert!(evaluate_diagram(&evolved_population.graph, diagram.root, &zero_bitvec));
        }
    }
}
