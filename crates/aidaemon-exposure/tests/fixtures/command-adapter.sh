#!/bin/sh
# Contract fixture for the `command` exposure adapter: announce the origin
# on the first stdout line, print more (ignored), then keep running.
port="$1"
service="$2"
case "$service" in
  client_gateway) name="client-gateway" ;;
  *) name="$service" ;;
esac
echo "{\"url\": \"https://${name}-${port}.example.com\"}"
echo "later output is ignored"
exec sleep 60
