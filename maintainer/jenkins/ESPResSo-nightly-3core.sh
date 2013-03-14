#!/bin/bash --login -e
source maintainer/jenkins/common.sh

bootstrap

start "CONFIGURE"
./configure --with-mpi CPU_COUNT="3"
end "CONFIGURE"

# copy config file
if [ "$myconfig" != default ]; then
  use_myconfig $myconfig
fi

# create mympiexec.sh
echo 'exec mpiexec --bind-to-core $@' > mympiexec.sh
chmod +x mympiexec.sh

start "BUILD"
make -j 3
end "BUILD"

check
