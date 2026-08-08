# Training Feed-Forward Multi-Layer Perceptrons with the Secretary Bird Optimization Algorithm (SBOA)

This repository presents a metaheuristic-based training approach for feed-forward multi-layer perceptrons (MLPs) using the Secretary Bird Optimization Algorithm (SBOA).

In this framework, the weights and biases of the MLP are encoded into candidate solution vectors, and the training process is formulated as an optimization problem. Instead of relying only on gradient-based learning methods such as backpropagation, SBOA is used to search for suitable network parameters.

The main motivation of this study is to provide an alternative training strategy for MLPs, especially in cases where gradient-based methods may suffer from issues such as local minima, sensitivity to initialization, and convergence difficulties.

## Related Paper

Burak Dilber, A. Fırat Özdemir (2026).  
**A novel approach to training feed-forward multi-layer perceptrons with recently proposed secretary bird optimization algorithm**.  
*Neural Computing and Applications*.  

- **Article page:** https://link.springer.com/article/10.1007/s00521-026-11874-x  
- **DOI:** https://doi.org/10.1007/s00521-026-11874-x

## Summary

- SBOA is adapted to train feed-forward MLPs.
- All trainable parameters of the network are optimized simultaneously.
- The proposed approach is evaluated on regression, function approximation, and classification problems.
- The results are compared with several well-known metaheuristic algorithms.

## Citation

If you use this work, please cite:

```bibtex
@article{Dilber2026SBOAMLP,
  author  = {Burak Dilber and A. F{\i}rat {\"O}zdemir},
  title   = {A novel approach to training feed-forward multi-layer perceptrons with recently proposed secretary bird optimization algorithm},
  journal = {Neural Computing and Applications},
  year    = {2026},
  doi     = {10.1007/s00521-026-11874-x}
}
