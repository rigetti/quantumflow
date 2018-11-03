# Acknowledgements

QuantumFlow began life in late 2017 as an internal project at Rigetti Computing
to explore quantum machine learning on near-term quantum computers. Many 
people have contributed either directly or indirectly.

QuantumFlow's model of hybrid quantum-classical programming is taken
from Quil, Rigetti's quantum programming language detailed in *A Practical
Quantum Instruction Set Architecture*, written by Robert S. Smith, Michael J.
Curtis, and William J. Zeng [1]. Much code was borrowed from pyQuil and Rigetti's
Grove, originally written by Robert Smith, Will Zeng, and Spike Curtis, with
significant contributions from Anthony Polloreno, Peter Karalekas, Nikolas
Tezak, Chris Osborn, Steven Heidel, and Matt Harrigan (among others). The
quil parser is adapted from the python parser written by Steven Heidel.
QuantumFlow's latex generation code was inspired by Anthony Polloreno's quil
to latex module. Construction of the decomposition module was assisted by
Eric C. Peterson (Of ECP gate fame), and Josh Combes made contributions to
the measures module. Diamond norm was adapted from code originally written by
Marcus P. da Silva. Keri McKerinan and Chris M. Wilson were early beta
testers. And finally Nick Rubin undoubtably has the single greatest
contribution, as the gate definitions, Kraus operators, and simulation model
were all adapted from his reference-qvm [2].  
  

Gavin E. Crooks (2018-11-01)


   [1]  A Practical Quantum Instruction Set Architecture
        Robert S. Smith, Michael J. Curtis, William J. Zeng
        arXiv:1608.03355 https://arxiv.org/abs/1608.03355

   [2]  reference-qvm: A reference implementation for a quantum virtual machine
   	    in Python https://github.com/rigetticomputing/reference-qvm