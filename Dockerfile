FROM continuumio/anaconda3
LABEL maintainer="Guillaume SIMON"

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root
RUN apt-get update && apt-get -y install emacs make

WORKDIR /workdir
COPY Makefile /workdir/Makefile
COPY setup.py /workdir/setup.py
COPY tsl /workdir/tsl
COPY tsl_env.yml /workdir/tsl_env.yml
COPY tsl_config.yml /workdir/tsl_config.yml

ARG USER_ID
ARG GROUP_ID
RUN if [ ${USER_ID:-0} -ne 0 ] && [ ${GROUP_ID:-0} -ne 0 ]; then \
    groupadd -g ${GROUP_ID} ds &&\
    useradd -l -u ${USER_ID} -g ds ds &&\
    install -d -m 0755 -o ds -g ds /home/ds &&\
    install -d -m 0755 -o ds -g ds /workdir \	
;fi
RUN touch /opt/conda/envs/.conda_envs_dir_test
RUN chown ${USER_ID}:${GROUP_ID} /opt/conda/envs/.conda_envs_dir_test
RUN mkdir -p /opt/conda/pkgs
#RUN touch /opt/conda/pkgs/urls.txt && chown ${USER_ID}:${GROUP_ID} /opt/conda/pkgs/urls.txt
RUN chown -R ${USER_ID}:${GROUP_ID} /opt/conda/pkgs

USER ds
RUN conda init
RUN make conda-install

ENTRYPOINT ls -la
