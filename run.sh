#!/bin/bash
cd "$(dirname "$0")"

pip3 install -r requirements.txt
python3 code.py $1 $2 $3