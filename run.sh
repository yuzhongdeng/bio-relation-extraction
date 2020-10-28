#!/bin/bash
cd "$(dirname "$0")"

pip3 install transformers
pip3 install sentencepiece==0.1.91

pip3 install -r requirements.txt
python3 code.py $1 $2 $3