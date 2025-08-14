#!/bin/bash
# Usage: ./fetch_all.sh <TOKEN>

TOKEN=$1

if [ -z "$TOKEN" ]; then
    echo "Usage: ./fetch_all.sh <TOKEN>"
    exit 1
fi

CATEGORIES=("equitiesexpanded" "commoditiesexpanded" "bondsexpanded" "cryptoexpanded" "etfsexpanded")

for category in "${CATEGORIES[@]}"
do
    echo "Fetching $category..."
    python3 scripts/fetch_category.py "$TOKEN" "$category"
done
