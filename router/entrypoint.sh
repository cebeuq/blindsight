#!/bin/sh
# blindsight router entrypoint (POSIX sh). Runs in the pinned haproxy alpine image (uid 99, read-only filesystem):
# turns TEXT_URL into the variables router/haproxy.cfg uses, checks them, then hands over to the image's own
# entrypoint (haproxy -W -db: master-worker, foreground). A bad value: one line on stderr and exit 2, no HAProxy.
#   TEXT_URL       http(s)://host[:port][/prefix]   default http://host.docker.internal:8000
#                  host: DNS name, IPv4 or [IPv6] (no zone id; http only: HAProxy verifies certificates against
#                  DNS names, so https needs a name). Not allowed: user:password@ (the client's own API key headers
#                  pass through), ?query, #fragment. Trailing slashes of the prefix are dropped; the client's request
#                  path and query are appended to it.
#   TEXT_CA_FILE   optional PEM file inside the container: replaces the system CAs for an https TEXT_URL (it must
#                  be readable; set with an http TEXT_URL it is an error). Unset for haproxy.cfg when empty, so
#                  @system-ca applies.
#   TEXT_HOST / TEXT_PORT  v1 settings: must arrive empty (set, they are an error naming the TEXT_URL to use); the
#                  derived values below reuse the names.
# Derived and exported for haproxy.cfg:
#   TEXT_SCHEME    http | https
#   TEXT_HOST      host, lower case, IPv6 without brackets (also the TLS SNI name the certificate must match)
#   TEXT_PORT      port (default 80 / 443)
#   TEXT_ADDR      host:port for the server line ([IPv6]:port)
#   TEXT_PREFIX    path prefix: "" or "/a/b"
#   TEXT_PREFIX_FMT  the prefix escaped for HAProxy's log-format ("%" -> "%%"); set only when there is a prefix
#   TEXT_AUTHORITY Host header sent upstream: host, plus ":port" only when it is not the scheme's default
#   TEXT_DNS       static: an IP literal or a name in /etc/hosts (host.docker.internal via extra_hosts), resolved
#                  once at startup | runtime: re-resolved through Docker's DNS while running (hosted APIs rotate IPs)
# Error messages never echo the whole URL (it may hold a secret in the wrong place).
# Debug: docker compose run --rm --no-deps router env   (prints the derived TEXT_* instead of starting HAProxy)
set -eu

die() {
  printf 'blindsight router: configuration error: %s\n' "$1" >&2
  exit 2
}

# Dotted-quad IPv4 (each part 0-255, no leading zeros). A function so "set --" leaves the command line alone.
is_ipv4() {
  old_ifs=$IFS; IFS=.; set -- $1; IFS=$old_ifs
  [ $# -eq 4 ] || return 1
  for o; do
    case $o in ''|*[!0-9]*|0?*) return 1 ;; esac
    [ "${#o}" -le 3 ] && [ "$o" -le 255 ] || return 1
  done
}

# v1 settings: refuse them rather than silently using the TEXT_URL default (compose passes both, empty by default).
if [ -n "${TEXT_HOST:-}" ] || [ -n "${TEXT_PORT:-}" ]; then
  die "TEXT_HOST / TEXT_PORT are no longer used: replace them with TEXT_URL=http://${TEXT_HOST:-<host>}:${TEXT_PORT:-<port>} in .env (and delete both)"
fi

url=$(printf '%s' "${TEXT_URL:-}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
[ -n "$url" ] || url=http://host.docker.internal:8000
case $url in
  *[[:space:][:cntrl:]]*) die "TEXT_URL contains whitespace or a control character" ;;
  *://*) ;;
  *) die "TEXT_URL must look like http(s)://host[:port][/path], e.g. https://api.example.com/anthropic" ;;
esac

scheme=$(printf '%s' "${url%%://*}" | tr '[:upper:]' '[:lower:]')
rest=${url#*://}
case $scheme in
  http) default_port=80 ;;
  https) default_port=443 ;;
  *) die "TEXT_URL must start with http:// or https://" ;;
esac
case $rest in *'#'*) die "TEXT_URL may not have a #fragment" ;; esac
case $rest in *'?'*) die "TEXT_URL may not have a ?query (the client's own path and query are appended to it)" ;; esac
authority=${rest%%/*}
path=${rest#"$authority"}
case $authority in
  *@*) die "TEXT_URL may not contain user:password@ (the client's own API key headers are passed through)" ;;
  '') die "TEXT_URL has no host" ;;
esac

# host and port
v6=0
case $authority in
  \[*)
    inner=${authority#\[}
    host=${inner%%]*}
    [ "$host" != "$inner" ] || die "TEXT_URL: unclosed [ around the IPv6 address"
    after=${inner#*]}
    case $after in
      '') port= ;;
      :*) port=${after#:} ;;
      *) die "TEXT_URL: unexpected text after the ] of the IPv6 address" ;;
    esac
    case $host in *%*) die "TEXT_URL: IPv6 zone ids (%...) are not supported" ;; esac
    case $host in *:*) ;; *) die "TEXT_URL: [$host] is not an IPv6 address" ;; esac
    case $host in *[!0-9A-Fa-f:.]*) die "TEXT_URL: [$host] is not an IPv6 address" ;; esac
    v6=1 ;;
  *)
    host=${authority%%:*}
    port=${authority#"$host"}
    port=${port#:}
    case $port in *:*) die "TEXT_URL: put an IPv6 address in brackets, e.g. http://[fd00::1]:8000" ;; esac ;;
esac
host=$(printf '%s' "$host" | tr '[:upper:]' '[:lower:]')
[ -n "$host" ] || die "TEXT_URL has no host"
ip=$v6
if [ "$v6" = 0 ]; then
  case $host in
    *[!a-z0-9._-]*|.*|*.|*..*|-*) die "TEXT_URL: '$host' is not a valid host name or IP address" ;;
    *[!0-9.]*) ;;
    *) is_ipv4 "$host" || die "TEXT_URL: '$host' is not a valid IPv4 address"; ip=1 ;;
  esac
fi
if [ -n "$port" ]; then
  case $port in *[!0-9]*) die "TEXT_URL: the port must be a number" ;; esac
  while :; do case $port in 0?*) port=${port#0} ;; *) break ;; esac; done
  { [ "${#port}" -le 5 ] && [ "$port" -ge 1 ] && [ "$port" -le 65535 ]; } || die "TEXT_URL: port $port is not in 1-65535"
else
  port=$default_port
fi

# path prefix: RFC 3986 path characters only (anything else must be percent-encoded), no trailing slashes
if printf '%s' "$path" | grep -q "[^A-Za-z0-9._~!\$&'()*+,;=:@/%-]"; then
  die "TEXT_URL: the path has a character that must be percent-encoded"
fi
if printf '%s' "$path" | grep -qE '%([^0-9A-Fa-f]|[0-9A-Fa-f][^0-9A-Fa-f]|[0-9A-Fa-f]?$)'; then
  die "TEXT_URL: the path has a % that is not followed by two hex digits"
fi
while :; do case $path in */) path=${path%/} ;; *) break ;; esac; done

if [ "$v6" = 1 ]; then hostpart="[$host]"; else hostpart=$host; fi
text_authority=$hostpart
[ "$port" = "$default_port" ] || text_authority="$hostpart:$port"

dns=runtime
if [ "$ip" = 1 ] || awk -v h="$host" '{ sub(/#.*/, ""); for (i = 2; i <= NF; i++) if (tolower($i) == h) f = 1 }
                                     END { exit !f }' /etc/hosts 2>/dev/null; then
  dns=static
fi

ca=${TEXT_CA_FILE:-}
tls=-
if [ "$scheme" = https ]; then
  # HAProxy checks the certificate against DNS names only (subjectAltName dNSName / CN), never IP entries, so an
  # IP literal could never verify: refuse it here rather than fail every request later.
  [ "$ip" = 0 ] || die "TEXT_URL: https needs a host name, not an IP address (the certificate is checked against DNS names). Name the address with extra_hosts in a compose override and use a certificate for that name"
  tls="verify, ca=${ca:-system}"
  if [ -n "$ca" ] && ! { [ -f "$ca" ] && [ -r "$ca" ]; }; then
    die "TEXT_CA_FILE=$ca is not a readable file inside the router container (mount it with a compose override, see .env.example)"
  fi
elif [ -n "$ca" ]; then
  die "TEXT_CA_FILE is set but TEXT_URL is http, so nothing would be verified: use an https TEXT_URL or unset TEXT_CA_FILE"
fi
case $host in
  localhost|127.*|::1)
    printf 'blindsight router: warning: TEXT_URL host %s is the router container itself; a model server on the Docker host is host.docker.internal\n' "$host" >&2 ;;
esac

export TEXT_SCHEME="$scheme" TEXT_HOST="$host" TEXT_PORT="$port" TEXT_ADDR="$hostpart:$port" TEXT_PREFIX="$path" \
  TEXT_AUTHORITY="$text_authority" TEXT_DNS="$dns"
if [ -n "$ca" ]; then export TEXT_CA_FILE="$ca"; else unset TEXT_CA_FILE; fi
if [ -n "$path" ]; then
  TEXT_PREFIX_FMT=$(printf '%s' "$path" | sed 's/%/%%/g')
  export TEXT_PREFIX_FMT
else
  unset TEXT_PREFIX_FMT
fi
printf 'blindsight router: text model %s://%s:%s%s (Host: %s, dns=%s, tls=%s)\n' \
  "$scheme" "$hostpart" "$port" "$path" "$text_authority" "$dns" "$tls" >&2

[ "$#" -gt 0 ] || set -- haproxy -f /usr/local/etc/haproxy/haproxy.cfg
exec docker-entrypoint.sh "$@"
