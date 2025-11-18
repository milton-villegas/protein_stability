# Bayesian Optimization Specification Document (Permanent Reference)

## 1. Project Purpose
This document defines the functional requirements for an adaptive experimental optimization system based on Bayesian Optimization (BO).  
The system is designed to support buffer screening and can be extended to broader DOE-like experimental workflows, including protein crystallization and other biophysical optimization tasks.

## 2. Core Objectives
- Use Bayesian Optimization to identify optimal experimental conditions based on user-defined objectives.
- Allow the user to specify any number of optimization targets (e.g., maximizing Tm, minimizing aggregation, maximizing affinity, optimizing kinetic constants).
- Ensure the BO model uses only the variables explicitly requested by the user.
- Maintain compatibility with multiple experimental platforms:
  - nanoDSF (e.g., Tm)
  - Creoptix WAVE (e.g., aggregation, affinity, kon, koff)
  - Future analytical or crystallization methods

## 3. Data Structure Requirements
All experimental results are provided in a table format.  
Each column has an explicit header, which identifies a measurable property that can be used in BO.

Examples of common headers:
- `Tm`
- `Aggregation`
- `KD`
- `kon`
- `koff`
- User-defined future metrics

The BO system must:
1. Validate that requested optimization objectives match existing headers.  
2. Dynamically configure the objective function(s) based on the selected headers.  
3. Support an arbitrary number of objectives (“multi-objective optimization”).

## 4. Optimization Logic Requirements
The BO system must adapt the objective function depending on the user request:

### 4.1 Single-objective cases
- Example: *maximize Tm*
- Example: *minimize aggregation*

### 4.2 Multi-objective cases
- Example: *minimize aggregation AND maximize affinity*
- Example: *maximize Tm AND minimize aggregation AND optimize kinetics*
- The system must combine objectives into a coherent BO strategy (e.g., scalarization or Pareto approaches).

### 4.3 Flexibility
- The system must not be restricted to a fixed set of metrics.
- Any header in the table can become an optimization target.
- The number of objectives is not limited.

## 5. Platform Adaptability
The system must be inherently modular so that the same BO logic can be applied to:
- Buffer screening experiments
- Protein stability assays
- Binding assays
- Aggregation assays
- Crystallization condition optimization (future)
- DOE-style multi-factor experiments

This adaptability must be maintained even if new assays or metrics are introduced.

## 6. Role of the Model Using This Document (Claude)
The model must always:
1. Refer back to this document when generating recommendations, code, analysis, or experimental strategies.
2. Ensure that all outputs are consistent with the project purposes and BO requirements defined here.
3. Verify that user-requested objectives:
   - exist in the dataset,
   - are valid measurable properties,
   - and are integrated correctly into the BO logic.
4. Provide suggestions or corrections if an optimization request conflicts with this specification.
5. Maintain coherence with the long-term vision of the system:  
   an open-source, extensible experimental optimization platform that can function similarly to commercial DOE tools.

## 7. Extensibility Requirements
The system must be prepared for future expansions such as:
- Additional metrics
- New equipment outputs
- More complex multi-objective optimization workflows
- Crystallization condition optimization
- Integration with automated experimentation systems

The document should remain stable and backward compatible.

