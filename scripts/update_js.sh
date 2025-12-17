#!/usr/bin/env bash
# Copyright lowRISC contributors (OpenTitan project).
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Find the Git repository root directory
GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)

if [ -z "$GIT_ROOT" ]; then
    echo "Error: This script must be run from within a Git repository."
    exit 1
fi

# Define the target directory relative to the Git root
TARGET_DIR="$GIT_ROOT/src/dvsim/templates/static"

# Create the directory (and any missing parents) if it doesn't exist
mkdir -p "$TARGET_DIR/css" "$TARGET_DIR/js"
echo "Downloading latest Bootstrap and htmx to $TARGET_DIR..."

# Bootstrap CSS (latest 5.x)
curl -L -o "$TARGET_DIR/css/bootstrap.min.css" \
    https://cdn.jsdelivr.net/npm/bootstrap@5/dist/css/bootstrap.min.css

# Bootstrap JS Bundle (includes Popper, latest 5.x)
curl -L -o "$TARGET_DIR/js/bootstrap.bundle.min.js" \
    https://cdn.jsdelivr.net/npm/bootstrap@5/dist/js/bootstrap.bundle.min.js

# htmx (latest version via @latest tag)
curl -L -o "$TARGET_DIR/js/htmx.min.js" \
    https://cdn.jsdelivr.net/npm/htmx.org@latest/dist/htmx.min.js

echo "Done! Files saved to:"
echo "  - $TARGET_DIR/css/bootstrap.min.css"
echo "  - $TARGET_DIR/js/bootstrap.bundle.min.js"
echo "  - $TARGET_DIR/js/htmx.min.js"
