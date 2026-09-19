"""Check relative Markdown link targets against actual source-distribution members."""

import posixpath
import re
import sys
import tarfile
from urllib.parse import unquote, urlsplit


def check(path):
    missing = []
    with tarfile.open(path) as archive:
        members = {m.name: m for m in archive.getmembers()}
        for name, member in members.items():
            if not member.isfile() or not name.endswith(".md"):
                continue
            text = archive.extractfile(member).read().decode("utf-8")
            text = re.sub(r"```.*?```", "", text, flags=re.S)
            links = re.findall(r"\]\(([^\s)]+)(?:\s+[^)]*)?\)", text)
            links += re.findall(r"^\s*\[[^]]+\]:\s*(\S+)", text, flags=re.M)
            for link in links:
                url = urlsplit(link.strip("<>"))
                if url.scheme or url.netloc or not url.path:
                    continue
                target = posixpath.normpath(
                    posixpath.join(posixpath.dirname(name), unquote(url.path))
                )
                if target not in members and not any(
                    n.startswith(target.rstrip("/") + "/") for n in members
                ):
                    missing.append(f"{name}: {link}")
    if missing:
        raise SystemExit("Missing sdist documentation targets:\n" + "\n".join(missing))
    print("All relative Markdown targets are included in the sdist")


if __name__ == "__main__":
    check(sys.argv[1])
