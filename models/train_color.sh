#!/usr/bin/env sh
set -e

./home/dirie/deep_learning/caffe/build/tools/caffe train --solver=home/dirie/cnn_practice/models/color_solver.prototxt $@
