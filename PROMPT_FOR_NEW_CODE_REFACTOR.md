I want to start a new project. This project will be about meta-learning Pandora's box problems for the application of LLM inference-time optimization.
This project will require code that:
1. Trains and saves neural networks on datasets
2. Runs algorithms/policies using those trained neural networks as well as with baselines and saves those results
3. Makes plots of the results using the saved data of optimization or whatever from step 2, as well as of training the NNs from step 1.
Anyway, the specific project doesn't matter, but what matters is that observe that this code will have the same overall structure as this project: in this project, step 1 is training the acquisition function neural networks, step 2 is running the BO loops, and step 3 is making the plots of everything.
Therefore, it would be great if I could make the code for my new project use some of the same functionality as this project. Such functionality includes for example things like automatic SLURM job submissions, YAML config files, neural network model training, saving, and loading, and automatic plot generation scripts (and potentially more things I couldn't think of).
But the problem is that this codebase we have here is very specific to this project, and it is quite complex and has hacky workarounds and aspects specific to this project. Some aspects are over-engineered and I would want to re-do.
Ideally, I would want to re-use the core functionality from this project in the new project's code.
- One option would be to try to *add onto the existing codebase* so that it can also run experiments for this new project, but I think this would make the code too complex, fragmented, and unmaintainable.
- I think a better idea would be to *abstract out all appropriate reusable aspects* so the existing code is refactored in a more modular way into perhaps a python module? and then *use the same shared functionality to build a new repo from scratch*.
- Or perhaps instead of a python module, maybe could keep it in the same git repository, where at the base it has the shared functionality, and have two sub-directories, one for the BO project and the other for the new project...need to think about this.

Your task is to write a proposal for how exactly to go about doing this. I want you to produce a .md file detailing the proposal.

In order to take on this undertaking, I think the following steps should be followed:
1. Understand at a high level how this existing code works (like what are the different modules/components and how do they all fit together)
2. Identify which specific parts of my existing code that I want to reuse.
3. Make the plan of how to do this.
