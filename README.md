# Boilerplate MLFlow

Take on mlflow starter for a ease of getting into MlOps

## Installing

### Pienv & pyenv

Make sure you have `pyenv` installed. Please follow the [Pyenv repository](https://github.com/pyenv/pyenv) to install it. Use [Pyenv-win reposytory](https://github.com/pyenv-win/pyenv-win) if you're on Windows

Make sure you have `pipenv` installed. Please follow the [Pienv documentation](https://pipenv-fork.readthedocs.io/en/latest/install.html) to install it.

**Pay attention** if you are on Windows and didn't add `pipenv` to your `PATH` you need always add `python -m` in front of `pipenv`

### Setup the environment
- clone this repository
- navigate to the root of the repository
- from the root of the repository run `pipenv install` in case of errors remove `Pipfile.lock` and try again
- activate the environment by `pipenv shell`
- run `python entry_training.py` to run a training
- run `mlflow ui` to confirm that a run was tracked by mlflow

### Git hooks (optional)

**Pay attention** this step is optional but strongly suggested, as it make sure that linter is applyed

Make sure that `pre-commit` is installed
Run `pre-commit install` to setup all the from `.pre-commit-config.yaml`

### Notebook

To run a notebook, please make sure youre get into your virtual environtment `pipenv shell`
Install named kernel `python -m ipykernel install --user --name=boilerplate-mlflow`
After you run `jupyter notebook` from virtual environment and select a kernel named `boilerplate-mlflow`



## Next steps

### How to think about the next steps

- Implement your own preprocessing
- Implement your own modeling
- Connect to a remote mlFlow server
- Start experimenting
- Converge on hyperparmenters and features
- Use mlflow to run your inference
