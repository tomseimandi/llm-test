#! /bin/bash
mkdir $HOME/opt && cd $HOME/opt
# Download a supported openssl version. e.g., openssl-1.1.1o.tar.gz or openssl-1.1.1t.tar.gz
wget https://www.openssl.org/source/openssl-1.1.1o.tar.gz
tar -zxvf openssl-1.1.1o.tar.gz
cd openssl-1.1.1o
./config && make && make test
mkdir $HOME/opt/lib
mv $HOME/opt/openssl-1.1.1o/libcrypto.so.1.1 $HOME/opt/lib/
mv $HOME/opt/openssl-1.1.1o/libssl.so.1.1 $HOME/opt/lib/

export LD_LIBRARY_PATH=$HOME/opt/lib:$LD_LIBRARY_PATH
