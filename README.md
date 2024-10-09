The pupose of this repo was for an academic assignment for the course Evolutionary Computing at the Vrije Universiteit Amsterdam. The majority of the code is a clone from https://github.com/karinemiras/evoman_framework

In our code, we implemented three algorithms to train on EvoMan as a specialist. Moreover, we implemented tracing of the statistics during the run, and asserting performance of the best models. 

First, to install the enviroment, install enviroment.yml using conda. Be advised that this has more dependencies than the orignal EvoMan, as we also use CMA-ES and NEAT packages.

Now, to train the three algorithms on enemies 1, 2, and 3 with standard parameters, in the main directory run:
- optimization_NEAT_specialist.py
- optimization_GA_specialist.py
- optimization_CMA-ES_specialist.py
If you want to use different parameters, this can easily be done by changing the parameters as flags. 


Now to get all info for the runs, additionaly run
- neat_info_extract.py
- neat_information_gain.py

Now everything should be ready to get results of plots and t test by running:
- make_plots.py
