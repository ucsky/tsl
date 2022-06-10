.PHONY: help

SHELL:=bash -i

help:
	@grep -E '(^[a-zA-Z0-9_-]+:.*?##.*$$)' Makefile | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[32m%-30s\033[0m %s\n", $$1, $$2}'

# Enable BuildKit for Docker build
export DOCKER_BUILDKIT:=1

define conda_c_or_u
	(conda env list | grep tsl >> /dev/null || conda env $(1) -f tsl_env.yml) \
	&& conda activate tsl && python setup.py install \
	&& pip install jupyterlab==3.4.2 \
	&& pip install ipywidgets==7.7.0 \
	&& pip install neptune-client==0.16.3 \
	&& jupyter nbextension enable --py widgetsnbextension
endef


tsl_config.yaml: Makefile
	touch $@ \
	&& echo "neptune_token: $$NEPTUNE_API_TOKEN" >> $@ \
	&& echo "neptune_username: $$NEPTUNE_API_USERNAME" >> $@

conda-install: ## Install environment.
conda-install:
	$(call conda_c_or_u,create)

conda-update: ## Update conda environment.
conda-update:
	$(call conda_c_or_u,update)

conda-remove: ## Remove conda environment.
conda-remove:
	conda env remove -n tsl

conda-startlab: ## Start jupyterlab.
conda-startlab:
	export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
	&& conda activate tsl \
	&& jupyter lab --no-browser

DOCKER_INAME:=tsl
docker-build: ## Build docker image.
docker-build: tsl_config.yaml
	docker build \
	--build-arg USER_ID=$$(id -u $$USER) \
	--build-arg GROUP_ID=$$(id -g $$USER) \
	--tag $(DOCKER_INAME) . 

DOCKER_CNAME:=tsl
docker-run-it: ## Run docker image.
docker-run-it:
	docker run \
	-it \
	--entrypoint /bin/bash \
	--rm  \
	-v `pwd`:/workdir \
	--shm-size=4gb \
	--name $(DOCKER_CNAME)  $(DOCKER_INAME)
#
IMPUTATION:=examples/imputation/run_imputation.py
CFG_IMPUTATION_GRIN:=examples/imputation/config/grin.yaml
CFG_IMPUTATION_RNNI:=examples/imputation/config/rnni.yaml
CFG_IMPUTATION_TEST:=examples/imputation/config/test.yaml

#	| yq '.hidden_size = 1' \
#	| yq '.ff_size = 2' \

#
$(CFG_IMPUTATION_TEST): $(CFG_IMPUTATION_GRIN) Makefile
	cat $< \
	| yq '.window = 1' \
	| yq '.epochs = 4' \
	| yq '.batches_per_epoch = 8' \
	| yq '.batch_size = 2' \
	> $@

# -m cProfile -o output.pstats
test-imputation: ## Testing imputation.
test-imputation: $(CFG_IMPUTATION_TEST)
	if [ ! -f  tsl_config.yaml ];then make tsl_config.yaml;fi \
	&& export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
	&& conda activate tsl \
	&& python $(IMPUTATION) \
	--dataset-name re_small \
	--config test.yaml \
	--neptune-logger \
	--workers 16 \
# 	
#	&& export CUDA_LAUNCH_BLOCKING=1 \
#	&& export CUDA_VISIBLE_DEVICES="" \
#
