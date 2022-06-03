FROM continuumio/anaconda3
WORKDIR /workdir
COPY Makefile /workdir/
COPY tsl_env.yml /workdir/
COPY setup.py /workdir/
RUN apt-get update && apt-get install emacs make
RUN make conda-install
ENTRYPOINT ls -la