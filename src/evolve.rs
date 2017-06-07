#![allow(unused_variables)]
use diagram::Diagram;
use rand::Rng;

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

fn generate_branch<R>(rng: &mut R, params: RDDParams, diagram: &mut Diagram) -> usize
    where R: Rng
{
    let variable = rng.gen_range(0, params.variable_count);
    let low = rng.gen_range(0, diagram.len());
    let high = rng.gen_range(0, diagram.len());
    if low == high {
        low
    } else {
        diagram.branch(variable, low, high)
    }
}

struct StrategySelector {
    strategy: EvolutionStrategy,
    pending_children: Vec<(usize, f64)>,
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

    fn choose_parents<R>(&mut self, rng: &mut R, roots: &[usize]) -> (usize, usize)
        where R: Rng
    {
        let a_idx = rng.gen_range(0, roots.len());
        let mut b_idx = rng.gen_range(0, roots.len());
        while b_idx == a_idx {
            b_idx = rng.gen_range(0, roots.len());
        }
        let result = (a_idx, b_idx);
        self.pending_parents = Some(result);
        self.pending_children.clear();
        result
    }

    fn add_child(&mut self, child: usize, fitness: f64) {
        self.pending_children.push((child, fitness));
    }

    fn output_generation(&mut self, individuals: &mut Vec<usize>, fitness: &mut Vec<f64>) {
        let parents = self.pending_parents
            .take()
            .expect("Need to be in a generation to output it.");
        match (self.pending_children.get(0), self.pending_children.get(1)) {
            (Some(&(child_0, f_0)), Some(&(child_1, f_1))) if parents.0 != parents.1 => {
                individuals[parents.0] = child_0;
                fitness[parents.0] = f_0;
                individuals[parents.1] = child_1;
                fitness[parents.1] = f_1;
                for &(child, f) in &self.pending_children[2..] {
                    individuals.push(child);
                    fitness.push(f);
                }
            }
            (Some(&(child_0, f_0)), Some(&(child_1, f_1))) if parents.0 == parents.1 => {
                individuals[parents.0] = child_0;
                fitness[parents.0] = f_0;
                individuals.push(child_1);
                fitness.push(f_1);
            }
            (Some(&(child_0, f_0)), None) if parents.0 != parents.1 => {
                individuals[parents.0] = child_0;
                fitness[parents.0] = f_0;
                individuals.remove(parents.1);
                fitness.remove(parents.1);
            }
            (Some(&(child_0, f_0)), None) if parents.0 == parents.1 => {
                individuals[parents.0] = child_0;
                fitness[parents.0] = f_0;
            }
            (None, None) if parents.0 != parents.1 => {
                individuals.remove(parents.0);
                fitness.remove(parents.0);
                individuals.remove(parents.1);
                fitness.remove(parents.1);
            }
            (None, None) if parents.0 == parents.1 => {
                individuals.remove(parents.0);
                fitness.remove(parents.0);
            }
            _ => unreachable!(),
        }
    }
}

pub struct DiagramPopulation {
    diagram: Diagram,
    roots: Vec<usize>,
}

pub fn evolve_diagrams<'a, F, R>(rng: &'a mut R,
                                 es: EvolutionStrategy,
                                 params: RDDParams,
                                 fitness: F)
                                 -> DiagramPopulation
    where F: Fn(&Diagram, usize) -> f64,
          R: Rng
{
    let pop = population(es);
    let mut diagram = Diagram::with_capacity(pop);
    let mut roots = (0..pop)
        .map(|_| generate_branch(rng, params, &mut diagram))
        .collect::<Vec<usize>>();
    let mut fit = {
        roots
            .iter()
            .map(|root| fitness(&diagram, *root))
            .collect::<Vec<f64>>()
    };
    let mut strategy = StrategySelector::new(es);
    for _ in 0..generations(es) {
        let (a, b) = strategy.choose_parents(rng, &roots);
        let parent;
        let fitness_parent;
        let fitness_a = fit[a];
        let fitness_b = fit[b];
        if fitness_a >= fitness_b {
            parent = roots[a];
            fitness_parent = fitness_a;
        } else {
            parent = roots[b];
            fitness_parent = fitness_b;
        }
        let num_children = 2;
        for _ in 0..num_children {
            let mut child = parent;
            let mut adapted = false;
            for _ in 0..params.mutation_count {
                match rng.gen_range(0, 6) {
                    0 => {
                        if let Some(c) = mutate_n1(rng, child, &mut diagram) {
                            child = c;
                        }
                    }
                    1 => {
                        if let Some(c) = mutate_n1_inv(rng, child, &mut diagram) {
                            child = c;
                        }
                    }
                    2 => {
                        if let Some(c) = mutate_n2(rng, child, &mut diagram) {
                            child = c;
                        }
                    }
                    3 => {
                        if let Some(c) = mutate_n2_inv(rng, child, &mut diagram) {
                            child = c;
                        }
                    }
                    4 => {
                        if let Some(c) = mutate_n3(rng, child, &mut diagram) {
                            child = c;
                        }
                    }
                    5 => {
                        if let Some(c) = mutate_a1(rng, parent, &mut diagram) {
                            adapted = true;
                            child = c;
                        }
                    }
                    _ => unreachable!(),
                }
            }
            let fitness_child = if adapted {
                fitness(&diagram, child)
            } else {
                fitness_parent
            };
            strategy.add_child(child, fitness_child);
        }
        strategy.output_generation(&mut roots, &mut fit);
    }

    return DiagramPopulation { diagram, roots };
}

fn mutate_n1<R>(rng: &mut R, parent: usize, diagram: &mut Diagram) -> Option<usize> {
    return None;
}

fn mutate_n1_inv<R>(rng: &mut R, parent: usize, diagram: &mut Diagram) -> Option<usize> {
    return None;
}

fn mutate_n2<R>(rng: &mut R, parent: usize, diagram: &mut Diagram) -> Option<usize> {
    return None;
}

fn mutate_n2_inv<R>(rng: &mut R, parent: usize, diagram: &mut Diagram) -> Option<usize> {
    return None;
}

fn mutate_n3<R>(rng: &mut R, parent: usize, diagram: &mut Diagram) -> Option<usize> {
    return None;
}

fn mutate_a1<R>(rng: &mut R, parent: usize, diagram: &mut Diagram) -> Option<usize> {
    return None;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::XorShiftRng;

    #[test]
    fn evolve_can_find_0() {
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
        let fitness = |diagram: &_, root| if root == 0 {
            return 1.0f64;
        } else {
            return 0.0f64;
        };
        let evolved_population = evolve_diagrams(&mut rng, strategy, params, fitness);
        for root in &evolved_population.roots {
            assert_eq!(*root, 0);
        }
    }
}
