#!/bin/bash
# Usage: ./fetch_all.sh <TOKEN>

TOKEN=$1

if [ -z "$TOKEN" ]; then
    echo "Usage: ./fetch_all.sh <TOKEN>"
    exit 1
fi

CATEGORIES=("equitiesexpanded" "commoditiesexpanded" "cryptoexpanded")

for category in "${CATEGORIES[@]}"
do
    echo "Fetching $category..."
    python3 scripts/fetch_category.py "$TOKEN" "$category"
    sleep 4  # polite delay between categories
done
