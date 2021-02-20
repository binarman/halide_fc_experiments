#!/bin/bash
for i in build/n*; do
#  echo $i | cut -f 2 -d/
  cp runner.cpp $i
  cp HalideBuffer.h $i
  cp HalideRuntime.h $i
  pushd $i > /dev/null
  g++ runner.cpp fc.o -ldl -lpthread -o fc
  ./fc
  popd > /dev/null
done
