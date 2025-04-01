#!/bin/bash
LTIME=`stat -c %Z ""$1".tex"`
pdflatex ""$1".tex"
zathura ""$1".pdf" &
while true    
do
    ATIME=`stat -c %Z ""$1".tex"`
 
    if [[ "$ATIME" != "$LTIME" ]]
    then    
        pdflatex -halt-on-error -interaction=nonstopmode ""$1".tex"; 
        biber $1
        pdflatex -halt-on-error -interaction=nonstopmode ""$1".tex"; 
        echo $?
        PDFTIME=`stat -c %Z ""$1".pdf"`
        if [[ PDFTIME > ATIME ]] 
        then
            if ! pgrep -x "zathura" > /dev/null
            then
               zathura ""$1".pdf" &
            fi
        else  
            enscript ""$1".log" --output=""$1"_log.ps"
            ps2pdf ""$1"_log.ps" ""$1"_log.pdf"
            zathura ""$1"_log.pdf" &
        fi 
        LTIME=`stat -c %Z ""$1".tex"`
    fi
    sleep 1
done

