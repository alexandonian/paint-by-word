#!/usr/bin/env bash
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

mkdir -p lib && cd lib && git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git