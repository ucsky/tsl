SHELL := bash -i

help:
	@grep -E '(^[a-zA-Z0-9_-]+:.*?##.*$$)' Makefile | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[32m%-30s\033[0m %s\n", $$1, $$2}'


conda-install: ## Install.
conda-install:
	(conda env list | grep tsl >> /dev/null || conda env create -f tsl_env.yml) \
	&& conda activate tsl && python setup.py install \
	&& pip install jupyterlab \
	&& pip install ipywidgets \
	&& jupyter nbextension enable --py widgetsnbextension


conda-startlab: ## Start jupyterlab.
conda-startlab:
	export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
	&& conda activate tsl \
	&& jupyter lab --no-browser

DOCKER_INAME := tsl
docker-build: ## Build docker image.
docker-build:
	docker build --tag $(DOCKER_INAME) . 


DOCKER_CNAME := tsl
docker-run-it: ## Run docker image.
docker-run-it:
	docker run -it --entrypoint /bin/bash --rm -v `pwd`:/workdir --name $(DOCKER_CNAME)  $(DOCKER_INAME)

IMPUTATION := examples/imputation/run_imputation.py
# 	&& export CUDA_VISIBLE_DEVICES="" \
test-imputation: ## Testing imputation.
test-imputation:
	export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
	&& conda activate tsl \
	&& python $(IMPUTATION) --epochs 1 --dataset-name air36
