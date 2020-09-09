#!/bin/bash
CC=dpcpp
PROJECT=myproject
source /opt/intel/inteloneapi/setvars.sh
# source /opt/intel/inteloneapi/setvars.sh --dnnl-configuration=cpu_gomp --force> /dev/null 2>&1

CFLAGS="-O3 -fpic -std=c++11"
LDFLAGS="-L${DNNLROOT}/lib"
INCFLAGS="-I${DNNLROOT}/include"
GLOB_ENVS="-DDNNL_CPU_RUNTIME=SYCL -DDNNL_GPU_RUNTIME=SYCL"
PROJECT=myproject

# ${CC} ${CFLAGS} ${INCFLAGS} -c firmware/model.cpp -o model.o ${LDFLAGS} -ldnnl 
# ${CC} ${CFLAGS} ${INCFLAGS} -shared model.o -o firmware/model.so ${LDFLAGS} -ldnnl 
${CC} ${CFLAGS} ${INCFLAGS} -c firmware/${PROJECT}.cpp -o ${PROJECT}.o ${LDFLAGS} -ldnnl 
${CC} ${CFLAGS} ${INCFLAGS} -shared ${PROJECT}.o -o firmware/${PROJECT}.so ${LDFLAGS} -ldnnl 
rm -f *.o