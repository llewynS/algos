# AlgOS: Algorithm Operating System

Algorithm Operating System (AlgOS) is an unopinionated, extensible, modular framework for algorithmic implementations. AlgOS offers numerous features: integration with Optuna for automated hyperparameter tuning; automated argument parsing for generic command-line interfaces; automated registration of new classes; and a centralised database for logging experiments and studies. These features are designed to reduce the overhead of implementing new algorithms and to standardise the comparison of algorithms. The standardisation of algorithmic implementations is crucial for reproducibility and reliability in research. AlgOS combines Abstract Syntax Trees with a novel implementation of the Observer pattern to control the logical flow of algorithmic segments. 

## Installation
Install via pip:
```sh
pip install algos_core
```

Install from cloned repository:

```sh
pip install -e .
```

Use in python via 
```python
import algos
```

## Modules
AlgOS is designed to be extensible for any algorithm that can be constructed as a cyclic or acyclic graph. 

Currently the available modules are:

* Reinforcement Learning: [AlgOSRL](https://github.com/llewynS/algosrl)

The intention is to expand this in the future. If you use this library and would like to include a module please let us know. 

## Examples
The tests contain some examples of how to construct basic algorithms using AlgOS. [AlgOSRL](https://github.com/llewynS/algosrl) contains concrete examples of how to construct reinforcement learning algorithms using AlgOS.

## Documentation
TBA

## Citing the Project

To cite this repository in publications:

```bibtex
@misc{algos,
      title={AlgOS: Algorithm Operating System}, 
      author={Llewyn Salt and Marcus Gallagher},
      year={2025},
      eprint={2504.04909},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2504.04909}, 
}
```
