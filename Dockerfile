FROM continuumio/anaconda3
WORKDIR /workdir
COPY Makefile /workdir/
COPY README.md /workdir/
COPY tsl_env.yml /workdir/
COPY setup.py /workdir/
COPY tsl /workdir/
RUN apt-get update && apt-get -y install emacs make
RUN make conda-install
ENTRYPOINT ls -la