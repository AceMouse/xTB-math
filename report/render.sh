#!/usr/bin/env -S nix shell nixpkgs#texliveFull nixpkgs#zathura nixpkgs#ghostscript_headless nixpkgs#enscript nixpkgs#inotify-tools --command bash

mainfile="$1"

# Initial compile
pdflatex -halt-on-error -interaction=nonstopmode "$mainfile.tex"
biber "$mainfile"
pdflatex -halt-on-error -interaction=nonstopmode "$mainfile.tex"

zathura "$mainfile.pdf" &

# Watch for changes in .tex files in current directory and subdirectories
while inotifywait -r -e modify,create,delete,move --exclude '\.swp$' .; do
    echo "Change detected. Recompiling..."

    if pdflatex -halt-on-error -interaction=nonstopmode "$mainfile.tex"; then
        biber "$mainfile"
        pdflatex -halt-on-error -interaction=nonstopmode "$mainfile.tex"
    else
        mainfile_log="${mainfile}_log"
        echo "Compilation failed. Showing log..."
        enscript "$mainfile.log" --output="$mainfile_log.ps"
        ps2pdf "$mainfile_log.ps" "$mainfile_log.pdf"
        zathura "$mainfile_log.pdf" &
    fi
done
