# pyRecEst

*Recursive Bayesian Estimation for Python*

pyRecEst is a Python library tailored for recursive Bayesian estimation, compatible with numpy, pytorch, and jax backends.

Features of pyRecEst include:

* Distribution and Densities: Provides tools for handling distributions and densities across Euclidean spaces and manifolds.
* Filters and Trackers: Offers a suite of recursive Bayesian estimators (filters or trackers) for both Euclidean spaces and manifolds. This includes capabilities for:
  * Multi-Target Tracking (MTT)
  * Extended Object Tracking (EOT)
* Evaluation Framework: Contains an evaluation framework to facilitate comparison between different filters.
* Sampling Methods: Includes methods for sampling of the distributions and generating grids.

## Usage

Please refer to the test cases for usage examples.

## Citation

If you use **pyRecEst** in your research, please cite:

<table>
  <tr>
    <th>BibTeX</th>
    <th>BibLaTeX</th>
  </tr>
  <tr>
    <td>
      <pre><code class="language-bibtex">@misc{pfaff_pyRecEst_2023,
  author       = {Florian Pfaff},
  title        = {pyRecEst: Recursive Bayesian Estimation for Python},
  year         = {2023},
  howpublished = {\url{https://github.com/FlorianPfaff/pyRecEst}},
  note         = {MIT License}
}</code></pre>
    </td>
    <td>
      <pre><code class="language-biblatex">@software{pfaff_pyRecEst_2023_software,
  author    = {Florian Pfaff},
  title     = {pyRecEst: Recursive Bayesian Estimation for Python},
  year      = {2023},
  url       = {https://github.com/FlorianPfaff/pyRecEst},
  license   = {MIT},
  keywords  = {Bayesian filtering; manifolds; tracking; Python; NumPy; PyTorch; JAX}
}</code></pre>
    </td>
  </tr>
</table>

## Credits

- Florian Pfaff (<pfaff@ias.uni-stuttgart.de>)

pyRecEst borrows its structure from libDirectional and follows its code closely for many classes. libDirectional, a project to which I contributed extensively, is [available on GitHub](https://github.com/libDirectional). The backend implementations are based on those of [geomstats](https://github.com/geomstats/geomstats).

## License
`pyRecEst` is licensed under the MIT License.
