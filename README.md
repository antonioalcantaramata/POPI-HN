# POPI-HN: Pareto Optimal Prediction Intervals with Hypernetworks

POPI-HN is a hypernetwork-based model for probabilistic forecasting, which can obtain a complete set of solutions for the coverage-width trade-off (Pareto front) of Prediction Intervals. POPI-HN does not require tuning additional parameters, and it has been designed to deal with deep networks.

Pytorch HN structure has been adapted from [https://github.com/AvivNavon/pareto-hypernetworks](https://github.com/AvivNavon/pareto-hypernetworks)

## Usage

Three .py files can be found in the framework folder: models, solvers, and utils. The first contains the hypernetwork and target network structures, while the others are needed for the training process. Furthermore, a Python notebook has been added as a using example. 


## Citation and Funding

If you use this code, you can cite our paper:

Alcantara, A., Galvan, I. M., & Aler, R. (2023). Pareto Optimal Prediction Intervals with Hypernetworks. Applied Soft Computing, 133, 109930. https://doi.org/10.1016/j.asoc.2022.109930

Funded by PID2019-107455RB-C22/AEI/10.13039/501100011033.
Probabilistic Prediction and Metaheuristic Optimization of Solar/Wind Resources in the Iberian Peninsula (MET4LOWCAR / PROB-META). Agencia Estatal de Investigación.
