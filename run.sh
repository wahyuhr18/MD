#!/bin/sh

export BLOCKSIZE=64
export OMP_NUM_THREADS=1

  # date
  # echo "md.x running..."
  # ./md.x
  # date

# lists=(1024 2048 4096 8192 16384 32768)
#2048 4096 8192 16384 32768 65536 131072
lists=(1024 2048 4096 8192) 


for list in ${lists[@]}; do

cat > md.in << __END
### # of ATOM
${list}
### # of STEP
100
### time step
0.01
__END

  echo "Natom",${list}
  echo "VELOCITY VERLET SERIAL velet.x running...: omp_num_threads="${OMP_NUM_THREADS}
  ts=`date +%s.%3N`
  ./velet.x
  te=`date +%s.%3N`
  tdiff=`echo "${te} - ${ts}" |bc -l`
  echo "run time = ${tdiff} (sec)"
  echo "**************************************************************************"
  echo "LEAPFROG SERIAL leap.x running...: omp_num_threads="${OMP_NUM_THREADS}
  ts=`date +%s.%3N`
  ./leap.x
  te=`date +%s.%3N`
  tdiff=`echo "${te} - ${ts}" |bc -l`
  echo "run time = ${tdiff} (sec)"
  echo "**************************************************************************"
  echo "VELOCITY VERLET PARALEL veletcu.x  running...: blocksize"${BLOCKSIZE}
  ts=`date +%s.%3N`
  ./veletcu.x ${BLOCKSIZE}
  te=`date +%s.%3N`
  tdiff=`echo "${te} - ${ts}" |bc -l`
  echo "run time = ${tdiff} (sec)"
  echo "**************************************************************************"
  echo "LEAPFROG PARALEL leapcu.x running...: blocksize"${BLOCKSIZE}
  ts=`date +%s.%3N`
  ./leapcu.x ${BLOCKSIZE}
  te=`date +%s.%3N`
  tdiff=`echo "${te} - ${ts}" |bc -l`
  echo "run time = ${tdiff} (sec)"
  echo "**************************************************************************"

done
