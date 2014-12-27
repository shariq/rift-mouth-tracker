#!/bin/bash

while true; do
 ./bin/webcam
 nano webcam.cpp
 git add .
 git commit
 make
done
