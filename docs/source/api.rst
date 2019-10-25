DEERpredict API reference
=========================

.. automodule:: DEERpredict
    :members:
    :show-inheritance:

The module is composed of three main classes, each for the prediction of different probe-based methods.
:py:mod:`DEERPrediction <DEERpredict.DEERPrediction>` performs the Double Electron-Electron Resonance prediction,
:py:mod:`PREPrediction <DEERpredict.PREPrediction>` performs Paramagnetic Relaxation Enhancement calculations, and
:py:mod:`PREPrediction <DEERpredict.FRETPrediction>` performs the Fluorescence Resonance Energy Transfer calculation.

:py:class:`Operations <DEERpredict.utils.Operations>` is the base class containing the class attributes to be
inherited by the calculation classes, as well as common methods to be used by the calculation classes.

New libraries should be defined on the :py:data:`DEERpredict.libraries.LIBRARIES` dictionary.

.. automodule:: DEERpredict.DEERPrediction
    :members:
    :show-inheritance:


.. automodule:: DEERpredict.PREPrediction
    :members:
    :show-inheritance:


.. automodule:: DEERpredict.FRETPrediction
    :members:
    :show-inheritance:

“Operations” class
------------------

.. autoclass:: DEERpredict.utils.Operations
    :members:

“RotamerLibrary” class
----------------------

.. autodata:: DEERpredict.libraries.LIBRARIES
    :annotation: Loaded from data/libraries.yaml

.. autoclass:: DEERpredict.libraries.RotamerLibrary
    :members:

Lennard-Jones parameters
------------------------

.. automodule:: DEERpredict.lennardjones

.. autodata:: DEERpredict.lennardjones.lj_parameters
    :annotation: = {atomtype: {'vdw', 'p_q', 'p_Rmin2', 'eps'}, ...}
