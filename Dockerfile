FROM continuumio/anaconda3
WORKDIR /workdir
COPY . /workdir/
RUN apt-get update && apt-get -y install emacs make
RUN make conda-install
ENTRYPOINT ls -la