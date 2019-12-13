#!/bin/bash

if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! [ -x "$(command -v brew)" ]; then
        echo "You're on mac and don't have 'brew' installed. That's ok, but some versions of MacOSx have portaudio prerequisites.";
        echo "If running fails try intalling brew";
    else
        $(brew install portaudio)
    fi
fi
