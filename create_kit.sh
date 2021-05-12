#!/bin/bash

zip -r reinforcement.zip reinforcement --exclude *.idea* --exclude *.git* --exclude *.pyc -q
echo "If there were no errors, the kit has been created in reinforcement.zip"

