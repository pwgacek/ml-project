#!/bin/bash

if [ -z "$1" ]; then
    echo "Use: $0 <filename fragment>"
    exit 1
fi

pattern="$1"

for file in ./${pattern}*.*; do
    if [ -f "$file" ]; then
        echo "=== $file ==="
        tail -n 1 "$file"
        echo
    fi
done
