"""Download only the pinned official WiSARD sample; verify before atomic rename."""

import argparse
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from experiments.real_image_environment.wisard import SIZE, verify

URL = "https://drive.google.com/uc?export=download&id=1uSgMXuZGVCrWM_151UcykyHejxxaVHOo"


class Confirmation(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}
        self.active = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.active = True
            self.action = attrs.get("action")
        if tag == "input" and self.active and attrs.get("type") == "hidden":
            self.fields[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.active = False


def fetch(destination):
    if destination.exists():
        verify(destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            raise ValueError(
                "expected official confirmation page; inspect changed download protocol"
            )
        parser = Confirmation()
        parser.feed(response.read(100_000).decode())
    if parser.action != "https://drive.usercontent.google.com/download":
        raise ValueError("unexpected download endpoint")
    if parser.fields.get("id") != "1uSgMXuZGVCrWM_151UcykyHejxxaVHOo":
        raise ValueError("unexpected Drive file id")
    url = parser.action + "?" + urllib.parse.urlencode(parser.fields)
    temporary = destination.with_suffix(".download")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("xb") as stream:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > SIZE:
                    raise ValueError("download exceeds pinned size")
                stream.write(chunk)
        verify(temporary)
        temporary.rename(destination)
    except Exception:
        # Retain partial bytes for inspection; do not overwrite a prior attempt.
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    fetch(p.parse_args().output)
