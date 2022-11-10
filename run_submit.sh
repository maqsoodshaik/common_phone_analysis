#!/bin/bash

singularity exec --nv --bind /data/corpora:/corpora,/data/users/maqsood/hf_cache:/cache /nethome/mmshaik/Hiwi/common_phone_analysis/common_phone.sif bash /nethome/mmshaik/Hiwi/common_phone_analysis/submit.sh \
    2> /data/users/maqsood/logs/${JOB_ID}.err.log \
    1> /data/users/maqsood/logs/${JOB_ID}.out.log
