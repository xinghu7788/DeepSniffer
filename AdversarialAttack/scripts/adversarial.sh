#!/bin/sh
cd ..
python AdversarialAttack.py
cp attack_result/*.log ../../Results/Table10/logs/
mkdir ../../Results/AdversarialAttack
cp attack_result/*.log ../../Results/AdversarialAttack/
