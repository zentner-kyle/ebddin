#![allow(unused_variables)]
use bit_vec::BitVec;
use diagram::{Branch, Graph, Node, OrderedDiagram};
use evaluate_diagram;
use evolution_strategies::{Engine, Population, PopulationIterMut, Problem, Strategy};
use rand::Rng;
use random::choose_from_iter;
use render::render_diagram;
use std::cmp::{Ordering, PartialOrd};
use std::collections::HashSet;
use std::collections::hash_map::{Entry, HashMap};
use std::fs::File;
use std::sync::Arc;
use walk::{visit_paths_postorder, visit_paths_preorder};

fn write_diagram(name: &str, diagram: &OrderedDiagram, graph: &Graph) {
    let mut f = File::create(name).unwrap();
    render_diagram(&mut f, diagram.clone(), graph).unwrap();
}

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

fn check_order(graph: &Graph, diagram: &OrderedDiagram) {
    for visit in visit_paths_postorder(diagram, graph) {
        let this_variable_index = if let Node::Branch { variable, .. } =
            graph.expand(visit.node) {
            diagram.variable_index(variable)
        } else {
            diagram.order.len()
        };
        for Branch { variable, .. } in visit.path.iter().map(|node| graph.expand_branch(*node)) {
            assert!(diagram.variable_index(variable) < this_variable_index);
        }
    }
}

#[derive(Clone, Debug)]
pub struct DiagramIndividual {
    fitness: f64,
    generation: usize,
    diagram: OrderedDiagram,
}

pub struct DiagramProblem<F, G>
    where F: FnMut(&Graph, &OrderedDiagram) -> f64,
          G: FnMut(usize) -> Option<(BitVec, bool)>
{
    variable_count: usize,
    graph: Graph,
    test_cases: Vec<(BitVec, bool)>,
    fitness: F,
    test_generator: G,
    generations: usize,
}

impl<F, G> DiagramProblem<F, G>
    where F: FnMut(&Graph, &OrderedDiagram) -> f64,
          G: FnMut(usize) -> Option<(BitVec, bool)>
{
    fn compute_fitness(&mut self, diagram: &OrderedDiagram) -> f64 {
        let mut fitness = (self.fitness)(&self.graph, diagram);
        for &(ref input, output) in &self.test_cases {
            if evaluate_diagram(&self.graph, diagram.root, input) != output {
                fitness -= 1.0;
            }
        }
        return fitness;
    }

    fn gc_graph(&mut self, population: PopulationIterMut<DiagramIndividual>) {
        let mut new_graph = Graph::new();
        let mut replacements: HashMap<usize, usize> = [(0usize, 0usize), (1usize, 1usize)]
            .iter()
            .cloned()
            .collect();
        for individual in population {
            for visit in visit_paths_postorder(&individual.diagram, &self.graph) {
                let replacement = if let Node::Branch {
                           variable,
                           low,
                           high,
                       } = self.graph.expand(visit.node) {
                    let new_low = *replacements
                                       .get(&low)
                                       .expect("should have already visited due to postorder");
                    let new_high = *replacements
                                        .get(&high)
                                        .expect("should have already visited due to postorder");
                    match replacements.entry(visit.node) {
                        Entry::Vacant(entry) => {
                            let replacement = new_graph.branch(variable, new_low, new_high);
                            entry.insert(replacement);
                            replacement
                        }
                        Entry::Occupied(entry) => *entry.get(),
                    }
                } else {
                    visit.node
                };
                if visit.node == individual.diagram.root {
                    individual.diagram.root = replacement;
                }
            }
        }
        self.graph = new_graph;
    }

    fn add_new_test_cases(&mut self, mut population: Population<DiagramIndividual>) {
        let mut new_test_cases = Vec::new();
        'outer: for &DiagramIndividual {
                         fitness,
                         ref diagram,
                         ..
                     } in population.iter() {
            if fitness == 0.0 {
                for _ in 0..10000000usize {
                    if new_test_cases.len() >= 1 {
                        //if new_test_cases.len() >= 5 {
                        break 'outer;
                    }
                    if let Some((input, output)) = (self.test_generator)(self.test_cases.len()) {
                        if evaluate_diagram(&self.graph, diagram.root, &input) != output {
                            new_test_cases.push((input, output));
                        }
                    } else {
                        break 'outer;
                    }
                }
            } else {
                break 'outer;
            }
        }
        if new_test_cases.len() > 0 {
            println!("Increasing number of test cases to {}",
                     self.test_cases.len() + new_test_cases.len());
        }
        self.test_cases.extend_from_slice(&new_test_cases);
        for individual in population.iter_mut() {
            individual.fitness = self.compute_fitness(&individual.diagram);
        }
    }

    fn add_initial_test_cases(&mut self, count: usize) {

        for _ in 0..count {
            if let Some((input, output)) = (self.test_generator)(self.test_cases.len()) {
                self.test_cases.push((input, output));
            } else {
                break;
            }
        }
    }
}

impl<F, G> Problem for DiagramProblem<F, G>
    where F: FnMut(&Graph, &OrderedDiagram) -> f64,
          G: FnMut(usize) -> Option<(BitVec, bool)>
{
    type Individual = DiagramIndividual;
    fn initialize<R>(&mut self, count: usize, rng: &mut R) -> Vec<Self::Individual>
        where R: Rng
    {

        (0..count)
            .map(|_| {
                let mut order: Vec<_> = (0..self.variable_count).collect();
                rng.shuffle(&mut order);
                let diagram = OrderedDiagram {
                    root: 0,
                    order: Arc::new(order),
                };
                let fitness = self.compute_fitness(&diagram);
                DiagramIndividual {
                    generation: 0,
                    diagram,
                    fitness,
                }
            })
            .collect()
    }

    fn mutate<R>(&mut self, child: &mut Self::Individual, rng: &mut R) -> bool
        where R: Rng
    {
        let mut needs_fitness_update = false;
        {
            let graph = &mut self.graph;
            child.generation += 1;
            #[cfg(debug_assertions)]
            write_diagram("before.dot", &child.diagram, graph);
            debug!("order before = {:#?}", child.diagram.order);
            let weights = [1.05, 1.0, 1.05, 1.0, 0.0, 8.0];
            let total: f64 = weights.iter().sum();
            let choice = rng.next_f64() * total;
            if choice <= weights[0..1].iter().sum() {
                debug!("n1");
                mutate_n1(rng, &mut child.diagram, graph);
            } else if choice <= weights[0..2].iter().sum() {
                debug!("n1_inv");
                mutate_n1_inv(rng, &mut child.diagram, graph);
            } else if choice <= weights[0..3].iter().sum() {
                debug!("n2");
                mutate_n2(rng, &mut child.diagram, graph);
            } else if choice <= weights[0..4].iter().sum() {
                debug!("n2_inv");
                mutate_n2_inv(rng, &mut child.diagram, graph);
            } else if choice <= weights[0..5].iter().sum() {
                debug!("n3");
                mutate_n3(rng, &mut child.diagram, graph);
            } else if choice <= weights[0..6].iter().sum() {
                debug!("a1");
                if mutate_a1(rng, &mut child.diagram, graph) {
                    needs_fitness_update = true;
                }
            } else {
                unreachable!();
            }
            debug!("order after = {:#?}", child.diagram.order);
            #[cfg(debug_assertions)]
            {
                write_diagram("after.dot", &child.diagram, graph);
                check_order(graph, &child.diagram);
                let reachable = reachable_set(graph, &child.diagram);
                let nodes = node_set(graph, &child.diagram);
                assert_eq!(reachable, nodes);
            }
        }
        if needs_fitness_update {
            child.fitness = self.compute_fitness(&child.diagram);
        }
        true
    }

    fn compare<R>(&mut self,
                  a: &Self::Individual,
                  b: &Self::Individual,
                  _rng: &mut R)
                  -> Option<Ordering>
        where R: Rng
    {
        let generation_cmp = a.generation.cmp(&b.generation);
        return Some(a.fitness
                        .partial_cmp(&b.fitness)
                        .map_or(generation_cmp, |c| c.then(generation_cmp)));
    }

    fn maintain<R>(&mut self, mut population: Population<Self::Individual>, rng: &mut R)
        where R: Rng
    {
        self.gc_graph(population.iter_mut());
        self.add_new_test_cases(population);
    }
}

pub fn evolve_diagrams<F, G, R>(rng: R,
                                strategy: Strategy,
                                variable_count: usize,
                                generations: usize,
                                fitness: F,
                                generator: G)
                                -> Engine<DiagramProblem<F, G>, R>
    where F: FnMut(&Graph, &OrderedDiagram) -> f64,
          G: FnMut(usize) -> Option<(BitVec, bool)>,
          R: Rng
{
    let graph = Graph::new();
    let mut problem = DiagramProblem {
        variable_count,
        graph,
        test_cases: Vec::new(),
        fitness,
        test_generator: generator,
        generations: 0,
    };
    problem.add_initial_test_cases(1);
    let mut engine = Engine::new(problem, strategy, rng);
    for i in 1..(1 + generations) {
        engine.run_generation();
        engine.mut_problem().generations += 1;
        if i % 50 == 0 {
            engine.maintain();
        }
        if i % 100 == 0 {
            write_diagram("checkpoint.dot",
                          &engine.fitest().diagram,
                          &engine.problem().graph);
            println!("Completed generation {}", i);
            println!("Best fitness = {}", engine.fitest().fitness);
            if engine.fitest().fitness == 0.0 {
                println!("Found optimal!");
                break;
            }
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

fn reachable_set(graph: &Graph, diagram: &OrderedDiagram) -> HashSet<usize> {
    visit_paths_postorder(diagram, graph)
        .map(|visit| visit.node)
        .collect()
}

fn node_set(graph: &Graph, diagram: &OrderedDiagram) -> HashSet<usize> {
    let mut result = HashSet::new();
    let mut frontier = vec![diagram.root];
    while let Some(node) = frontier.pop() {
        result.insert(node);
        if let Node::Branch { low, high, .. } = graph.expand(node) {
            frontier.push(low);
            frontier.push(high);
        }
    }
    return result;
}

fn make_parent_graph(graph: &Graph, diagram: &OrderedDiagram) -> HashMap<usize, Vec<usize>> {
    let mut parent_graph = HashMap::new();
    for visit in visit_paths_preorder(diagram, graph) {
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
    for visit in visit_paths_preorder(diagram, graph) {
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
        let mut inserted = false;
        if let Entry::Vacant(entry) = replacements.entry(node) {
            let replacement = graph.branch(variable, low_replacement, high_replacement);
            entry.insert(replacement);
            inserted = true;
        }
        if inserted {
            add_parents_if_ready(node,
                                 &graph,
                                 &mut ready,
                                 &parent_graph,
                                 &replacements,
                                 &ancestor_set);
        }
    }
    return *replacements.get(&diagram.root).unwrap_or(&diagram.root);
}

fn mutate_n1<R>(rng: &mut R, diagram: &mut OrderedDiagram, graph: &mut Graph) -> bool
    where R: Rng
{
    // Remove a random redundant test.
    if let Some((to_fix, replacement)) =
        choose_from_iter(rng,
                         visit_paths_preorder(diagram, graph).filter_map(|path| {
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
                         visit_paths_preorder(diagram, graph).filter_map(|visit| {
            let next_variable_idx;
            if let Node::Branch { variable, .. } = graph.expand(visit.node) {
                let next_variable = variable;
                next_variable_idx = diagram.variable_index(next_variable);
            } else {
                next_variable_idx = diagram.order.len();
            }
            let first_allowed_variable_idx;
            if let Some(parent) = visit.path.last() {
                let Branch {
                    variable,
                    low,
                    high,
                } = graph.expand_branch(*parent);
                first_allowed_variable_idx = 1 + diagram.variable_index(variable);
            } else {
                first_allowed_variable_idx = 0;
            }
            if first_allowed_variable_idx < next_variable_idx {
                Some((visit, first_allowed_variable_idx, next_variable_idx))
            } else {
                None
            }
        })) {
        let variable_index = rng.gen_range(min_variable_idx, next_variable_idx);
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
    let parent_graph = make_parent_graph(graph, diagram);
    if let Some((a, b)) = choose_from_iter(rng,
                                           visit_paths_preorder(diagram, graph)
                                               .flat_map(|visit| {
        let mut overlaps = Vec::new();
        if let Some(parents) = parent_graph.get(&visit.node) {
            for (i, parent) in parents.iter().enumerate() {
                let branch = graph.expand_branch(*parent);
                for parent2 in &parents[0..i] {
                    if *parent != *parent2 && graph.expand_branch(*parent2) == branch {
                        overlaps.push((*parent, *parent2));
                    }
                }
            }
        }
        overlaps.into_iter()
    })) {
        diagram.root = rebuild_diagram(graph, diagram, a, b);
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
                         visit_paths_preorder(diagram, graph).filter_map(|path| {
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
    for visit in visit_paths_preorder(diagram, graph) {
        if let Node::Branch {
                   low,
                   high,
                   variable,
               } = graph.expand(visit.node) {
            if variable != first_variable {
                continue;
            }
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
                    continue;
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
                         visit_paths_preorder(diagram, graph).filter_map(|path| {
            if let Node::Branch {
                       low,
                       high,
                       variable,
                   } = graph.expand(path.node) {
                return Some((path, variable, low, high));
            }
            return None;
        })) {
        let target_variable_index = diagram.variable_index(target_variable);
        if let Some(target) = choose_from_iter(rng,
                                               visit_paths_preorder(diagram, graph)
                                                   .filter_map(|visit| {
            match graph.expand(visit.node) {
                Node::Branch { variable, .. } => {
                    let variable_index = diagram.variable_index(variable);
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
    use env_logger;
    use evaluate_diagram;
    use rand::SeedableRng;
    use rand::XorShiftRng;
    use render::{render_diagram, render_whole_graph};
    use std::fs::File;
    use std::fs::OpenOptions;
    use std::io::Write;

    fn write_diagram(name: &str, diagram: &OrderedDiagram, graph: &Graph) {
        let mut f = File::create(name).unwrap();
        render_diagram(&mut f, diagram.clone(), graph).unwrap();
    }

    #[test]
    fn evolve_can_evolve_1() {
        let _ = env_logger::init();
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
        let generator = |_| None;
        let engine = evolve_diagrams(rng, strategy, variable_count, 10, fitness, generator);
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
        let _ = env_logger::init();
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
        let generator = |_| None;
        let engine = evolve_diagrams(rng, strategy, variable_count, 10, fitness, generator);
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
        let _ = env_logger::init();
        let test_name = "evolve_can_evolve_two_bit_parity";
        let rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        let variable_count = 2;
        let strategy = Strategy::MuPlusLambda { mu: 1, lambda: 10 };
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
            return f;
        };
        let generator = |_| None;
        let engine = evolve_diagrams(rng, strategy, variable_count, 50, fitness, generator);
        let graph = &engine.problem().graph;
        {
            let mut f = File::create(format!("test_output/{}_graph.dot", test_name)).unwrap();
            render_whole_graph(&mut f, &graph).unwrap();
        }
        for (idx, diagram) in engine.population().map(|i| &i.diagram).enumerate() {
            let mut f = File::create(format!("test_output/diagram_{}_{}.dot", test_name, idx))
                .unwrap();
            render_diagram(&mut f, diagram.clone(), graph).unwrap();
            assert!((engine.problem().fitness)(graph, diagram) == 4.0);
        }
    }

    //#[test]
    fn evolve_can_evolve_k_bit_parity() {
        let _ = env_logger::init();
        let test_name = "evolve_can_evolve_k_bit_parity";
        let rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        let variable_count = 8;
        let strategy = Strategy::MuPlusLambda { mu: 1, lambda: 10 };
        let mut fitness_rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        let num_challenges = 10000;
        let bitvecs: Vec<BitVec> = (0..num_challenges)
            .map(|_| random_bitvec(&mut fitness_rng, variable_count))
            .collect();
        let fitness = |graph: &Graph, diagram: &OrderedDiagram| {
            let mut f = 0.0;
            for bitvec in &bitvecs {
                let set_bits: u32 = bitvec.blocks().map(|b| b.count_ones()).sum();
                let expected_output = set_bits % 2 == 0;
                if evaluate_diagram(graph, diagram.root, &bitvec) != expected_output {
                    f -= 1.0;
                }
            }
            return f;
        };
        let generator = |_| None;
        let mut engine = evolve_diagrams(rng, strategy, variable_count, 2500, fitness, generator);
        {
            let graph = &engine.problem().graph;
            {
                let mut f = File::create(format!("test_output/{}_graph.dot", test_name)).unwrap();
                render_whole_graph(&mut f, &graph).unwrap();
            }
            for (idx, diagram) in engine.population().map(|i| &i.diagram).enumerate() {
                let mut f = File::create(format!("test_output/diagram_{}_{}.dot", test_name, idx))
                    .unwrap();
                render_diagram(&mut f, diagram.clone(), graph).unwrap();
            }
        }
        {
            let mut best = engine.fitest().clone();
            let mut reduction_rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
            let mut i = 0;
            while i < 100 &&
                  (mutate_n1(&mut reduction_rng,
                             &mut best.diagram,
                             &mut engine.mut_problem().graph) ||
                   mutate_n2(&mut reduction_rng,
                             &mut best.diagram,
                             &mut engine.mut_problem().graph)) {
                i += 1;
            }
            write_diagram(&format!("test_output/best_{}.dot", test_name),
                          &best.diagram,
                          &engine.problem().graph);
        }
        for individual in engine.population() {
            assert_eq!(0.0, individual.fitness);
        }
    }

    fn evolve_can_evolve_parity(variable_count: usize) {
        let _ = env_logger::init();
        let test_name = format!("evolve_can_evolve_{}_bit_parity", variable_count);
        let rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        let strategy = Strategy::MuPlusLambda { mu: 1, lambda: 10 };
        let mut generator_rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        let fitness = |graph: &Graph, diagram: &OrderedDiagram| { return 0.0; };
        let generator = |_| {
            let input = random_bitvec(&mut generator_rng, variable_count);
            let set_bits: u32 = input.blocks().map(|b| b.count_ones()).sum();
            let output = set_bits % 2 == 0;
            return Some((input, output));
        };
        let mut engine =
            evolve_diagrams(rng, strategy, variable_count, 10000000, fitness, generator);
        {
            let graph = &engine.problem().graph;
            {
                let mut f = File::create(format!("test_output/{}_graph.dot", test_name)).unwrap();
                render_whole_graph(&mut f, &graph).unwrap();
            }
            for (idx, diagram) in engine.population().map(|i| &i.diagram).enumerate() {
                let mut f = File::create(format!("test_output/diagram_{}_{}.dot", test_name, idx))
                    .unwrap();
                render_diagram(&mut f, diagram.clone(), graph).unwrap();
            }
        }
        {
            let test_case_count = engine.problem().test_cases.len();
            let generations = engine.problem().generations;
            let mut best = engine.fitest().clone();
            let mut reduction_rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
            let mut i = 0;
            while i < 100 &&
                  (mutate_n1(&mut reduction_rng,
                             &mut best.diagram,
                             &mut engine.mut_problem().graph) ||
                   mutate_n2(&mut reduction_rng,
                             &mut best.diagram,
                             &mut engine.mut_problem().graph)) {
                i += 1;
            }
            write_diagram(&format!("test_output/best_{}.dot", test_name),
                          &best.diagram,
                          &engine.problem().graph);
            OpenOptions::new()
                .write(true)
                .truncate(false)
                .append(true)
                .open("test_output/parity_range_stats.csv")
                .unwrap()
                .write(format!("{},{},{}\n", variable_count, test_case_count, generations)
                           .as_bytes())
                .unwrap();
        }
        for individual in engine.population() {
            assert_eq!(0.0, individual.fitness);
        }
    }

    #[test]
    fn evolve_parity_range() {
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open("test_output/parity_range_stats.csv")
            .unwrap()
            .write("bits,test_cases,generations\n".as_bytes())
            .unwrap();
        for variable_count in 2..3 {
            evolve_can_evolve_parity(variable_count);
        }
    }

    //#[test]
    fn evolve_can_evolve_n_bit_parity() {
        let _ = env_logger::init();
        let test_name = "evolve_can_evolve_n_bit_parity";
        let rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        let variable_count = 3;
        let strategy = Strategy::MuPlusLambda { mu: 1, lambda: 10 };
        let mut fitness_rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        //let num_challenges = 100000;
        //let bitvecs: Vec<BitVec> = (0..num_challenges)
        //.map(|_| random_bitvec(&mut fitness_rng, variable_count))
        //.collect();
        let fitness = |graph: &Graph, diagram: &OrderedDiagram| {
            return 0.0;
            //let mut f = 0.0;
            //for bitvec in &bitvecs {
            //let set_bits: u32 = bitvec.blocks().map(|b| b.count_ones()).sum();
            //let expected_output = set_bits % 2 == 0;
            //if evaluate_diagram(graph, diagram.root, &bitvec) != expected_output {
            //f -= 1.0;
            //}
            //}
            //return f;
        };
        let generator = |_| {
            let input = random_bitvec(&mut fitness_rng, variable_count);
            let set_bits: u32 = input.blocks().map(|b| b.count_ones()).sum();
            let output = set_bits % 2 == 0;
            return Some((input, output));
        };
        //let mut engine = evolve_diagrams(rng, strategy, variable_count, 4600, fitness, generator);
        //let mut engine = evolve_diagrams(rng, strategy, variable_count, 5000, fitness, generator);
        let mut engine =
            evolve_diagrams(rng, strategy, variable_count, 10000000, fitness, generator);
        {
            let graph = &engine.problem().graph;
            {
                let mut f = File::create(format!("test_output/{}_graph.dot", test_name)).unwrap();
                render_whole_graph(&mut f, &graph).unwrap();
            }
            for (idx, diagram) in engine.population().map(|i| &i.diagram).enumerate() {
                let mut f = File::create(format!("test_output/diagram_{}_{}.dot", test_name, idx))
                    .unwrap();
                render_diagram(&mut f, diagram.clone(), graph).unwrap();
            }
        }
        {
            let mut best = engine.fitest().clone();
            let mut reduction_rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
            let mut i = 0;
            while i < 100 &&
                  (mutate_n1(&mut reduction_rng,
                             &mut best.diagram,
                             &mut engine.mut_problem().graph) ||
                   mutate_n2(&mut reduction_rng,
                             &mut best.diagram,
                             &mut engine.mut_problem().graph)) {
                i += 1;
            }
            write_diagram(&format!("test_output/best_{}.dot", test_name),
                          &best.diagram,
                          &engine.problem().graph);
        }
        for individual in engine.population() {
            assert_eq!(0.0, individual.fitness);
        }
    }


    #[test]
    fn can_mutate_n1_inv_above_leaf() {
        let _ = env_logger::init();
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
        let _ = env_logger::init();
        let mut rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        let mut graph = Graph::new();
        let mut diagram = OrderedDiagram::from_root(0, 1);
        assert!(mutate_n1_inv(&mut rng, &mut diagram, &mut graph));
        assert!(!mutate_n1_inv(&mut rng, &mut diagram, &mut graph));
        let mut f = File::create("test_output/can_mutate_n1_inv_at_leaf.dot").unwrap();
        render_diagram(&mut f, diagram.clone(), &graph).unwrap();
    }

    #[test]
    fn can_mutate_n1_inv_to_full_tree() {
        let _ = env_logger::init();
        let mut rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        let mut graph = Graph::new();
        let mut diagram = OrderedDiagram::from_root(0, 2);
        assert!(mutate_n1_inv(&mut rng, &mut diagram, &mut graph));
        assert!(mutate_n1_inv(&mut rng, &mut diagram, &mut graph));
        assert!(mutate_n1_inv(&mut rng, &mut diagram, &mut graph));
        assert!(!mutate_n1_inv(&mut rng, &mut diagram, &mut graph));
        write_diagram("test_output/can_mutate_n1_inv_to_full_tree.dot",
                      &diagram,
                      &graph);
    }

    #[test]
    fn can_mutate_n2_inv() {
        let _ = env_logger::init();
        let mut rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        let mut graph = Graph::new();
        let mut diagram = OrderedDiagram::from_root(0, 2);
        // Build a full tree.
        assert!(mutate_n1_inv(&mut rng, &mut diagram, &mut graph));
        assert!(mutate_n1_inv(&mut rng, &mut diagram, &mut graph));
        assert!(mutate_n1_inv(&mut rng, &mut diagram, &mut graph));
        // Merge equivalent middle nodes.
        mutate_n2(&mut rng, &mut diagram, &mut graph);
        // Unmerge equivalent middle nodes.
        assert!(mutate_n2_inv(&mut rng, &mut diagram, &mut graph));
        // Can't unmerge equivalent middle nodes again.
        assert!(!mutate_n2_inv(&mut rng, &mut diagram, &mut graph));
        write_diagram("test_output/can_mutate_n2_inv_at_leaf.dot",
                      &diagram,
                      &graph);
    }

    #[test]
    fn can_mutate_n3_2_bit_parity() {
        let _ = env_logger::init();
        let mut rng = XorShiftRng::from_seed([0xde, 0xad, 0xbe, 0xef]);
        let mut graph = Graph::new();
        let low = graph.branch(1, 0, 1);
        let high = graph.branch(1, 1, 0);
        let root = graph.branch(0, low, high);
        let mut diagram = OrderedDiagram::from_root(root, 2);
        write_diagram("test_output/can_mutate_n3_2_bit_parity_before.dot",
                      &diagram,
                      &graph);
        assert!(mutate_n3(&mut rng, &mut diagram, &mut graph));
        let mut f = File::create("test_output/can_mutate_n3_2_bit_parity.dot").unwrap();
        render_diagram(&mut f, diagram.clone(), &graph).unwrap();
    }
}
