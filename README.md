# Boilerplate MLFlow

Take on mlflow starter for a ease of getting into MlOps

## Installing

MLflow requires `conda` to be on the `PATH` for the projects feature.

### Pienv & pyenv

Make sure you have `pyenv` installed. Please follow the [Pyenv repository](https://github.com/pyenv/pyenv) to install it. Use [Pyenv-win reposytory](https://github.com/pyenv-win/pyenv-win) if you're on Windows

Make sure you have `pipenv` installed. Please follow the [Pienv documentation](https://pipenv-fork.readthedocs.io/en/latest/install.html) to install it.

**Pay attention** if you are on Windows and didn't add `pipenv` to your `PATH` you need always add `python -m` in front of `pipenv`

### Setup the environment
- clone this repository
- navigate to the root of the repository
- run `pipenv install` in the directory of the Pipfile
- activate the environment by `pipenv shell`

### Git hooks

Make sure that `pre-commit` is installed
Run `pre-commit install` to setup all the from `.pre-commit-config.yaml`


## Next steps

### How to think about the next steps

- Setup it as reusable project
- Log what you want to log 
- Define params with mlflow
- Use favors to deploy a model

### Reusable project

The project is defined by files `MLproject` and `conda.yaml`

### Remote tracking

### Artifact store
#### What will be used as an artifact store?

Choose one of the following and make sure you MLFlow Server can assess them:

- Amazon S3
- Azure Blob Storage
- Google Cloud Storage
- FTP server
- SFTP Server
- NFS
- HDFS
