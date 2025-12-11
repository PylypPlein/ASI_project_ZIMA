# satisfaction-prediction

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This is your new Kedro project with PySpark setup, which was generated using `kedro 1.0.0`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a [data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention)
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the files `tests/test_run.py` and `tests/pipelines/data_science/test_pipeline.py` for instructions on how to write your tests. Run the tests as follows:

```
pytest
```

You can configure the coverage threshold in your project's `pyproject.toml` file under the `[tool.coverage.report]` section.

## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. Install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `catalog`, `context`, `pipelines` and `session`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)

## Experiment Tracking with Weights & Biases (W&B)

During Sprint 2, experiment tracking was implemented using **Weights & Biases (W&B)**.

Each time the Kedro pipeline is executed (`kedro run`), model training and evaluation
results are automatically logged to W&B. Logged information includes:
- Model parameters (from the `params` configuration)
- Evaluation metrics (e.g., F1-score)

### How to enable W&B logging locally

1. **Install the W&B library:**
   ```bash
   pip install wandb
Log in to your W&B account:



wandb login
You can find your API key here: https://wandb.ai

Run the Kedro pipeline:


kedro run
After running the pipeline, all metrics and configuration details will appear
in the W&B project dashboard

### Production model selection

This project uses Weights & Biases (W&B) for experiment tracking, model comparison, and performance visualization.
All training runs executed in Sprint 3 were logged automatically via the Kedro pipeline (kedro run).

- Below is a full summary of the experiments, including:

- A table of model statistics (exported directly from W&B)

- A comparative ROC AUC chart for all trained models

- Conclusions about the best-performing model

### Experiment Table (W&B Export)

The table below contains the full export from Weights & Biases and summarizes
feature statistics calculated across all runs:

| importance | stddev | p_value | n | p99_high | p99_low |
|-----------|--------|---------|---|----------|---------|
| 0.09872003250711103 | 0.016755228908695293 | 0.00009586550324495861 | 5 | 0.13321928041198855 | 0.0642207846022335 |
| 0.04187322226737107 | 0.009057974179125303 | 0.0002471371554509706 | 5 | 0.060523716296697755 | 0.023222728238044388 |
| 0.03569687119057301 | 0.007143724808959489 | 0.000182607086418236 | 5 | 0.050405899308990296 | 0.020987843072155718 |
| 0.02494920763917119 | 0.0034981152837765764 | 0.000045184767208033546 | 5 | 0.03215187535755848 | 0.017746539920783896 |
| 0.006765542462413787 | 0.0037397805116591953 | 0.007768479250140015 | 5 | 0.014465802225179805 | -0.0009347173003522299 |
| 0.09713531084924829 | 0.01858075922030262 | 0.00015311928028094397 | 5 | 0.1353933508972356 | 0.058877270801260984 |
| 0.04205607476635516 | 0.009903338160872736 | 0.00034319972162596507 | 5 | 0.062447184918271725 | 0.021664964614438592 |
| 0.033543275091426274 | 0.00735399016536819 | 0.00026032685388436167 | 5 | 0.048685242484865145 | 0.018401307697987403 |
| 0.02584315318976029 | 0.005142133922242082 | 0.00017856163673749684 | 5 | 0.03643087800750041 | 0.015255428372020171 |
| 0.00867533522958146 | 0.003136484830876009 | 0.0017364845648419347 | 5 | 0.015133400847182458 | 0.0022172696119804646 |
| 0.07425843153189753 | 0.01375094709640515 | 0.0001348741715787137 | 5 | 0.10257182166168227 | 0.045945041402112786 |
| 0.04370174725721245 | 0.009212451755999725 | 0.00022355439448299507 | 5 | 0.06267031275243058 | 0.02473318176199432 |
| 0.03754571312474604 | 0.009064579522268074 | 0.0003778422454679754 | 5 | 0.05620980764689923 | 0.01888161860259285 |
| 0.017858594067452205 | 0.003732046264934245 | 0.00021612534888235409 | 5 | 0.0255429289093129 | 0.010174259225591512 |
| 0.007212515237708161 | 0.002964290219710391 | 0.002770318429405162 | 5 | 0.0133160297709367 | 0.001109000704479622 |
| 0.05985371800081292 | 0.009423700955376854 | 0.00007136462666747207 | 5 | 0.07925724851732278 | 0.040450187484303055 |
| 0.041609101991060625 | 0.009941454764244828 | 0.00036297305837521183 | 5 | 0.06207869475614777 | 0.02113950922597348 |
| 0.028443722064201715 | 0.009202252618970624 | 0.0011495009520586345 | 5 | 0.04739128739530987 | 0.009496156733093557 |
| 0.010341324664770557 | 0.0011524382609420466 | 0.000018204969860999455 | 5 | 0.01271421097243972 | 0.007968438357101393 |
| 0.004835432750914382 | 0.0035143028541957683 | 0.01852570632669658 | 5 | 0.01207143090053079 | -0.0024005653987020275 |

# run
uvicorn src.api.main:app --reload --port 8000

# test health
curl http://127.0.0.1:8000/healthz

# prediction
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
   "lp": 1001,
   "id": 1001,
   "Gender": "Male",
   "Customer Type": "Loyal Customer",
   "Age": 30,
   "Type of Travel": "Business travel",
   "Class": "Business",
   "Flight Distance": 200,
   "Inflight wifi service": 5,
   "Departure/Arrival time convenient": 5,
   "Ease of Online booking": 5,
   "Gate location": 5,
   "Food and drink": 5,
   "Online boarding": 5,
   "Seat comfort": 5,
   "Inflight entertainment": 5,
   "On-board service": 5,
   "Leg room service": 5,
   "Baggage handling": 5,
   "Checkin service": 5,
   "Inflight service": 5,
   "Cleanliness": 5,
   "Departure Delay in Minutes": 0,
   "Arrival Delay in Minutes": 0.0
  }'

# database
sqlite3 local.db 'select * from predictions limit 5;'
