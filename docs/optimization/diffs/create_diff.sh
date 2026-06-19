#!/bin/bash

while read -r file; do
    echo "git diff gpu_cpu_improvements -- $file"
    diff_name=$(basename "$file" .py).diff

    git diff gpu_cpu_improvements -- ../../../"$file" > "$diff_name"
done < <(git diff gpu_cpu_improvements --name-only | grep '.py')