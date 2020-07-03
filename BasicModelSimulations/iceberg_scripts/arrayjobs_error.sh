#!/bin/bash

for filename in sge_errors/*; do
	dos2unix $filename
	qsub $filename
done
