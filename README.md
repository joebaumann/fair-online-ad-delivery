# Fairness in Online Ad Delivery

> **Note**
> This repo contains the code to run the simulations presented in the FAccT'2024 paper titled [**Fairness in Online Ad Delivery**](https://doi.org/10.1145/3630106.3658980).
> 


## Setup

Set up pyenv virtual environment and install the required packages:
```
pyenv virtualenv 3.11.2 online-ad-simulation
pyenv activate online-ad-simulation
pip install -r requirements.txt
```

## Running simulations

Make bash script executable:
```
chmod +x run.sh
```
then make sure to:
- check to configuration of the different scenarios in the `scenario_config.py` file
- adjust the run.sh script by specifying the scenario to run, the number of cpu cores to use, etc.

Finally, run the simulation with the command:
```
./run.sh
```
This will run the simulation for the specified scenario and generate the plots.

If you want to, for example, regenerate the plots for scenario `A`, which is saved in the `results` directory, use the command:
```
python plot_simulation_results.py -p results -s A
```
If you want to generate the combined fairness-utility-tradeoff plot for the scenarios `A`, `B`, `C`, and `D`, which are all stored in the `results` directory, run:
```
python plot_simulation_results.py -p results -sc A B C D
```


## Citing

```
@inproceedings{baumann2024fairaddelivery,
  year={2024},
  title={Fairness in Online Ad Delivery},
  author={Baumann, Joachim and Sapiezynski, Piotr and Heitz, Christoph and Hannak, Aniko},
  booktitle={Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency},
  url={https://doi.org/10.1145/3630106.3658980},
  pages={1418–1432},
  series={FAccT '24}
}
```