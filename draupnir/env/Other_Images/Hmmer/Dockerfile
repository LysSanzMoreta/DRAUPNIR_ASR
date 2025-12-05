FROM ubuntu:24.04

USER root
RUN apt-get update && apt-get install -y wget
RUN apt-get install -y less
RUN apt-get install -y nano
RUN apt-get install -y build-essential
RUN wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz
RUN tar xf hmmer-3.4.tar.gz
RUN cd hmmer-3.4
RUN ls hmmer-3.4
RUN chmod +x hmmer-3.4/configure
RUN ./hmmer-3.4/configure
RUN make
RUN make install

