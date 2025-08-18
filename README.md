# Energy System Monitoring Agent


## Installation

Clone the repository with git

```bash
git clone https://github.com/flowcean/uc-energy.git
``` 

Create a virtual environment and activate it

```bash
cd uc-energy
python -m venv venv
source ./venv/bin/activate
```

Install the package in editable mode

```bash
python -m pip install -e .
```


## Run the Simulation

With virtual environment activated run

```bash
python run.py
``` 

On the first start, it will some time to download the datasets.
Afterwards, it will start the simulation.

Results will be saved in the `_outputs` directory.

You can control number of steps and timeout of the file watcher with

```bash
python run.py steps 15 timeout 35
```
