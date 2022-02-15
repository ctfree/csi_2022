#!/bin/bash
x=1
while [ $x -le 500 ]
do
  echo " $x "
  x=$(( $x + 1 ))
  python train.py
done