SHELL := bash -i

help:
	@grep -E '(^[a-zA-Z0-9_-]+:.*?##.*$$)' Makefile | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[32m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install.
install:
	(echo "Setup ..." \
	&& (conda env list | grep tsl >> /dev/null || conda env create -f tsl_env.yml) \
	&& conda activate tsl && python setup.py install \
	&& pip install jupyterlab \
	&& pip install ipywidgets \
	&& jupyter nbextension enable --py widgetsnbextension \
	)


start-lab: ## Start jupyterlab.
start-lab:
	export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python && conda activate tsl && jupyter lab --no-browser

