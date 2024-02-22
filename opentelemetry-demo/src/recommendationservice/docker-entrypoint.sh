#! /bin/bash
while read -r env; do export "$env"; done < /etc/environment 
printenv
opentelemetry-instrument python recommendation_server.py