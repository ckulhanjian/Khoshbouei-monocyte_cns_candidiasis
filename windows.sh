#!/bin/bash

for svg_file in plots/scatters/Inner/*.svg; do
    # Replace .svg extension with .wmf for output filename
    wmf_file="${svg_file%.svg}.wmf"
    inkscape "$svg_file" --export-type=wmf --export-filename="$wmf_file"
    echo "Converted: $svg_file -> $wmf_file"
done

for svg_file in plots/scatters/Outer/*.svg; do
    # Replace .svg extension with .wmf for output filename
    wmf_file="${svg_file%.svg}.wmf"
    inkscape "$svg_file" --export-type=wmf --export-filename="$wmf_file"
    echo "Converted: $svg_file -> $wmf_file"
done

for svg_file in plots/heatmaps/*.svg; do
    # Replace .svg extension with .wmf for output filename
    wmf_file="${svg_file%.svg}.wmf"
    inkscape "$svg_file" --export-type=wmf --export-filename="$wmf_file"
    echo "Converted: $svg_file -> $wmf_file"
done