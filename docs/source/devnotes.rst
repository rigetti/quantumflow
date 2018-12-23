.. _devnotes:

=================
Development Notes
=================

.. contents:: :local:

Please refer to the github repository https://github.com/rigetti/quantumflow for source code, and to submit issues and pull requests. Documentation is hosted at readthedocs https://quantumflow.readthedocs.io/ .

Installation
############

See the introduction for installation instructions.
The Makefile contains targets for various common development tasks::

	> make help

Testing
#######

Use pytest and tox for testing. Examples::

	> make test                                # Run all unittests on current backend
	> pytest tests/test_states.py::test_zeros  # Test a single method
	> make doctests                            # Test code snippets in documentation
	> make testall                             # Test all backends (Using tox)

Some tests will be skipped depending on the current backend.

The full tox testing suite should verifie against two consecutive versions of 
python, in order to ameliorate dependency hell.
(But at present [Nov 2018] we're stuck on python 3.6 only until tensorflow supports 3.7)

Test Coverage
#############

To report test coverage::

	> make coverage

Note that only the active backend should expect full coverage.

Some lines can't be covered. This happens, for instance, due to the conditional backend imports.
The comment ``# pragma: no cover`` can be used to mark lines that can't be covered. `


Delinting
#########

Delinting uses ``flake8`` ::

	> make lint


Documentation
#############

Documentation is complied with `sphinx <http://www.sphinx-doc.org/>`_ . :: 

	> make docs

Source is located at docs/source and output is located at 
`docs/build/html/index.html`

Documentation coverage report::

	> make doccover

Google style docstrings: http://www.sphinx-doc.org/en/stable/ext/example_google.html


Typecheck
#########
Python type hints are used to document types of arguments and return values::

	> make typecheck

The ``# type: ignore`` pragma can be used to supress typechecking if necessary.


Conventions
###########

- nb -- Abbreviation for number
- N -- number of qubits
- theta -- gate angle (In Bloch sphere or equivalent)
- t -- Number of half turns in Block sphere (quarter cycles) or equivalent. (theta = pi * t)
- bk -- backend module for interfacing with tensor package (``import quantumflow.backend as bk``)
- ket -- working state variable
- rho -- density variable
- chan -- channel variable
- circ -- circuit variable
- G -- Graph variable



Import Hierarchy
################
::

                    version
                      ^
                      |                 tensorflow      torch         numpy
    quantumflow --> config                  ^             ^             ^
              |       ^               ......:......       :             :
              |       |               :           :       :             :
              |---> backend ---> [tensorflowbk|eagerbk|torchbk] ---> numpybk
              |       ^
              |       |
              |---> qubits
              |       ^
              |       |
              |---> states
              |       ^
              |       |
              |----> ops
              |       ^
              |       |
              |---> gates
              |       ^
              |       |
              |---> stdgates
              |       ^
              |       |
              |---> channels
              |       ^
              |       |
              |---> circuits
              |       ^
              |       |
              |---> programs
              |       ^
              |       |
              \---> forest



Backends
########

Only the bare essential functionality has been implemented or imported for 
each backend. No doudt other methods could be added. Additional methods should
follow numpy's conventions (where appropriate), and need to be implemented for
each backend. (One of the backend unit tests checks that each backend claims 
to support every required method.)


GEC 2018