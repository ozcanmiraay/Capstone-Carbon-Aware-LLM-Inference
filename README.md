## Power Model Profiling & Carbon Emissions Estimation for LLM Inference 

### Project Details

The current focus is to develop a power and energy consumption model for LLM inference within the Vidur simulator. Specifically, I will measure and estimate the power usage across different hardware setups, LLM configurations, and inference scenarios. I will track these metrics dynamically and use the simulation output data to estimate carbon emissions associated with the model's usage.

After creating this energy estimation framework, I will validate the results by running inference requests on actual GPUs from TU Berlin's HPC Cluster, which will allow me to refine the power model iteratively for accuracy. This validation step is for building a generalizable power model for LLMs that could work well even for different, previously untested model configurations.

This framework aims to support energy and emissions tracking directly within Vidur, offering users insights into the environmental impact of LLM operations.

My final deliverable will be a research paper with an associated Github codebase detailing my findings and contributing a reliable energy consumption & carbon emissions framework to the literature. I am contributing directly to a simulation tool to help developers/researchers/industry professionals assess their carbon footprints before deploying systems for real-life inference scenarios.

### Code Details

#### colab-demos
- This folder will be regularly updated with experiment results' analysis and visualizations.

#### vidur-codes
- This folder keeps track of the code I modified in Vidur simulator to fit my research needs better.

#### vidur-experiments
- This folder contains the experiment scripts I wrote to run Vidur simulations with different hardware configurations and inference scenarios.

#### unit-tests
- This folder contains the test scripts I wrote to validate that code modifications work as intended.
