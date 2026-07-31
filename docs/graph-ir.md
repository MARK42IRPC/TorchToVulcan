# Graph IR 1.0

The canonical machine-readable definition is
[`schemas/graph-ir.schema.json`](../schemas/graph-ir.schema.json). This document
describes decisions and invariants that are awkward or impossible to express in
JSON Schema alone.

## Identity and references

Tensor IDs are the graph's data-flow identity. Node inputs and outputs contain
tensor IDs, while graph inputs and outputs select tensors at the model
boundary. A visual edge is derived by finding the node that produces a tensor
and the nodes that consume it.

Node IDs and tensor IDs are stable within a graph. Display names are not IDs and
may be changed by the user. Importers must create deterministic unique IDs when
the source model omits names or contains duplicates.

## Node order

The `nodes` array is topologically ordered. Every non-optional node input must
be one of:

- a graph input;
- a constant tensor with an external `data` reference;
- an output produced by an earlier node.

Each tensor has at most one producer. A node cannot overwrite a graph input.
Optional ONNX inputs are represented by `null` entries so positional operator
semantics are retained.

## Shapes

A dimension is represented as:

- a non-negative integer for a known dimension;
- a string for a symbolic dimension;
- `null` when the dimension is unknown.

An empty shape represents a scalar. Symbol names express equality only within
the current graph; bounds for dynamic compilation will be a separate compiler
input.

## Layout

Layout is an explicit string rather than an enum so later compiler versions can
introduce packed and backend-specific layouts without invalidating imported
graphs. Importers use `UNKNOWN` when the layout cannot be proven. Layout
transitions must eventually become explicit graph operations before kernel
selection.

## Constants

Large tensor payloads are never embedded in Graph IR JSON. A constant contains
a `data` object that points to a binary resource by URI, byte offset, and byte
length. Paths are resolved relative to the graph document unless a transport
layer provides a virtual resource resolver.

## UI state

The optional `ui` object stores presentation state only. Removing it must not
change model behavior. Compiler passes may preserve positions for unchanged
nodes but must never make semantic decisions from coordinates or collapsed
state.

## Versioning

The `schema_version` uses `major.minor` numbering:

- a minor revision may add optional fields;
- a major revision may change meaning or remove fields;
- readers reject unsupported major versions;
- migrations operate on serialized documents and are covered by fixtures.

Schema validation checks document shape. Compiler validation additionally
checks references, uniqueness, producer ownership, topological order, operator
contracts, shapes, and data types.

