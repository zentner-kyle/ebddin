EBDDIN
======

This is more-or-less a reimplementation of the paper "Evolving Binary Decision
Diagrams using Implicit Neutrality" (Richard M. Downing, 2005) in Rust.

The major differences relative to that paper are:
 - The use of an append-only multi-rooted BDD datastructure, to allow multiple
   BDDs to share the same memory. This significantly complicates mutation,
   without any real benefit since EBDDIN works well with small population
   sizes.
 - Bloat avoidance by slightly preferring reductive mutations to their
   inverses. Specifically, n1 and n2 are chosen 5% more frequently than n1' and
   n2'.
 - Difference evaluation is not used.

Neat features:
 - Tests
 - BDD rendering to GraphViz dot format.
 - Successfully reproduces most of the results of the paper. However, the
   largest function which has been reproduced is the nine-bit parity function.
