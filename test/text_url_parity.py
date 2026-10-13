"""TEXT_URL parity, imgfix side (test/run.sh): imgfix's own parser on every line of the table (argv[1]), printed
exactly like run.sh prints router/entrypoint.sh's result: <url>|ok|<scheme>|<host>|<port>|<prefix> or <url>|error.
Runs inside the imgfix image (imports /app/imgfix.py)."""
import sys

sys.path.insert(0, "/app")
import imgfix  # noqa: E402

with open(sys.argv[1], encoding="utf-8") as f:
    for line in f.read().split("\n")[:-1]:
        if line.startswith("#"):
            continue
        try:
            t = imgfix.parse_text_url(line)
        except ValueError:
            print(f"{line}|error")
        else:
            print(f"{line}|ok|{t.scheme}|{t.host}|{t.port}|{t.prefix}")
