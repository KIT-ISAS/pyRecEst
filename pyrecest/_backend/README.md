# _Backend Folder

This folder contains code from the Geomstats project, adjusted for pyRecEst by Florian Pfaff. The original version of Geomstats is authored by Nina Miolane et al., and is a Python package geared towards Riemannian Geometry in Machine Learning.

## Original Project Details

- **Title**: Geomstats: A Python Package for Riemannian Geometry in Machine Learning
- **Authors**: Nina Miolane, Nicolas Guigui, Alice Le Brigant, Johan Mathe, Benjamin Hou, Yann Thanwerdas, Stefan Heyder, Olivier Peltre, Niklas Koep, Hadi Zaatiti, Hatem Hajri, Yann Cabanes, Thomas Gerald, Paul Chauchat, Christian Shewmake, Daniel Brooks, Bernhard Kainz, Claire Donnat, Susan Holmes, Xavier Pennec
- **Journal**: Journal of Machine Learning Research, 2020, Vol. 21, No. 223, Pp. 1-9
- **URL**: [Geomstats Project](http://jmlr.org/papers/v21/19-027.html)

## License

This code is provided under the MIT License. A copy of the license can be found in this folder.

## Modifications

The code in this folder has been modified by Florian Pfaff to adapt it to pyRecEst.

## (Adapted) Usage Instructions

In order to expose a new backend function/attribute to the rest of the
codebase, it is necessary to add the name to the respective list in the
`BACKEND_ATTRIBUTES` dictionary in `pyrecest/_backend/__init__.py`.
This serves two purposes:

1. Define a clear boundary between backend interface and backend-internal code:
   Only functions/attributes which are used outside the backend should be made
   available to the rest of the codebase.
1. Guarantee each backend exposes the same attributes:
   When loading a backend, the backend importer verifies that a backend
   provides each attribute listed in the `BACKEND_ATTRIBUTES` dict.
   This way, we guarantee that unit tests fail during CI builds when a
   maintainer/contributor forgets to provide an implementation of a feature for
   a particular backend.
   If a feature cannot be supported for some reason, the function should raise
   a `NotImplementedError` for the time being.
