# POPI-HN: Pareto Optimal Prediction Intervals with Hypernetworks

POPI-HN is a hypernetwork-based model for probabilistic forecasting, which can obtain a complete set of solutions for the coverage-width trade-off (Pareto front) of Prediction Intervals. POPI-HN does not require tuning additional parameters, and it has been designed to deal with deep networks.

Pytorch HN structure has been adapted from [https://github.com/AvivNavon/pareto-hypernetworks](https://github.com/AvivNavon/pareto-hypernetworks)

## Usage

Three .py files can be found in the framework folder: models, solvers, and utils. The first one contains the hypernetwork and target network structures, while the other ones are needed for the training process. Furthermore, a python notebook has been added as an using example. 


## Citation

If you use this code, you can cite it in you work as:

```
@misc{Alcantara_POPI-HN,
    author = "Alc{\'a}ntara, Antonio",
    title = "POPI-HN",
    year = 2022,
    url = "https://github.com/antonioalcantaramata/POPI-HN/"
  }
```
