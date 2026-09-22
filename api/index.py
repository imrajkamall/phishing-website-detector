import os
import re
import socket
import ipaddress
from urllib.parse import urlparse

import joblib
import numpy as np
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE = os.path.join(BASE_DIR, "live_phishing_model.pkl")
THRESHOLD_FILE = os.path.join(BASE_DIR, "live_threshold.txt")

FEATURES = [
    'having_IP_Address', 'URL_Length', 'Shortining_Service',
    'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix',
    'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length',
    'Favicon', 'port', 'HTTPS_token', 'Request_URL', 'URL_of_Anchor',
    'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
    'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe',
    'age_of_domain', 'DNSRecord', 'web_traffic', 'Page_Rank',
    'Google_Index', 'Links_pointing_to_page', 'Statistical_report'
]

SHORTENERS = {
    'bit.ly', 'goo.gl', 't.co', 'tinyurl.com', 'ow.ly', 'is.gd',
    'buff.ly', 'adf.ly', 'cutt.ly', 'rb.gy', 'shorturl.at',
    'tiny.cc', 'lnkd.in', 'rebrand.ly'
}

app = FastAPI(title="Phishing Website Detection API", version="1.0")


class ScanRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, value):
        value = value.strip()

        if not value:
            raise ValueError("Please enter a website URL.")

        if len(value) > 2048:
            raise ValueError("URL is too long.")

        if not re.match(r"^https?://", value, re.I):
            value = "https://" + value

        parsed = urlparse(value)

        if not parsed.hostname:
            raise ValueError("Invalid website URL.")

        return value


_model = None
_threshold = None


def load_model():
    global _model, _threshold

    if _model is None:
        _model = joblib.load(MODEL_FILE)

        with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
            _threshold = float(f.read().strip())

    return _model, _threshold


def host_is_public(host):
    if not host:
        return False

    try:
        infos = socket.getaddrinfo(
            host,
            None,
            type=socket.SOCK_STREAM
        )

        addresses = {item[4][0] for item in infos}

        if not addresses:
            return False

        for addr in addresses:
            ip = ipaddress.ip_address(addr)

            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
            ):
                return False

        return True

    except Exception:
        return False


def normalize_url(url):
    if re.match(r"^https?://", url, re.I):
        return url

    return "https://" + url


def safe_fetch(url):

    current = normalize_url(url)

    headers = {
        "User-Agent":
        "Mozilla/5.0 (compatible; PhishingDetector/1.0; research-demo)"
    }

    session = requests.Session()
    session.max_redirects = 5

    for _ in range(6):

        parsed = urlparse(current)

        if (
            parsed.scheme not in ("http", "https")
            or not host_is_public(parsed.hostname or "")
        ):
            raise ValueError(
                "The destination is not a publicly reachable HTTP(S) website."
            )

        response = session.get(
            current,
            headers=headers,
            timeout=(4, 8),
            allow_redirects=False,
            stream=True
        )

        if (
            300 <= response.status_code < 400
            and response.headers.get("Location")
        ):

            next_url = requests.compat.urljoin(
                current,
                response.headers["Location"]
            )

            response.close()
            current = next_url

            continue

        content_type = response.headers.get(
            "content-type", ""
        ).lower()

        if "text/html" not in content_type and content_type:

            response.close()

            raise ValueError(
                "The URL did not return an HTML webpage."
            )

        data = b""

        for chunk in response.iter_content(
            chunk_size=65536
        ):

            data += chunk

            if len(data) >= 1_500_000:
                break

        encoding = response.encoding or "utf-8"

        status_code = response.status_code

        response.close()

        return (
            current,
            data.decode(encoding, errors="ignore"),
            status_code
        )

    raise ValueError("Too many redirects.")


def is_ip(host):

    try:
        ipaddress.ip_address(host.split(":")[0])
        return True

    except Exception:
        return False


def lexical_features(url):

    parsed = urlparse(url)

    host = parsed.hostname or ""

    host_parts = [
        x for x in host.split(".") if x
    ]

    features = {}

    features["having_IP_Address"] = (
        -1 if is_ip(host) else 1
    )

    length = len(url)

    features["URL_Length"] = (
        1 if length < 54
        else (0 if length <= 74 else -1)
    )

    host_lower = host.lower()

    features["Shortining_Service"] = (
        -1
        if (
            host_lower in SHORTENERS
            or any(x in host_lower for x in SHORTENERS)
        )
        else 1
    )

    features["having_At_Symbol"] = (
        -1 if "@" in url else 1
    )

    features["double_slash_redirecting"] = (
        -1 if "//" in url[8:] else 1
    )

    features["Prefix_Suffix"] = (
        -1 if "-" in host else 1
    )

    sub_count = max(
        0,
        len(host_parts) - 2
    )

    features["having_Sub_Domain"] = (
        1 if sub_count == 0
        else (0 if sub_count == 1 else -1)
    )

    features["SSLfinal_State"] = (
        1 if parsed.scheme.lower() == "https"
        else -1
    )

    features["Domain_registeration_length"] = 0
    features["Favicon"] = 1

    try:
        port = parsed.port
    except ValueError:
        port = None

    features["port"] = (
        -1
        if port not in (None, 80, 443)
        else 1
    )

    features["HTTPS_token"] = (
        -1
        if re.search(r"https", host, re.I)
        else 1
    )

    features["Abnormal_URL"] = 1

    return features


def analyze_page(html, final_url):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    host = (
        urlparse(final_url).hostname
        or ""
    ).lower()

    page = {}

    anchors = soup.find_all(
        "a",
        href=True
    )

    if anchors:

        external = 0

        for anchor in anchors:

            href = anchor.get(
                "href",
                ""
            ).strip()

            if href.startswith(
                ("http://", "https://")
            ):

                anchor_host = (
                    urlparse(href).hostname
                    or ""
                ).lower()

                if (
                    anchor_host
                    and anchor_host != host
                ):
                    external += 1

        ratio = external / len(anchors)

        page["URL_of_Anchor"] = (
            1 if ratio < 0.31
            else (0 if ratio < 0.67 else -1)
        )

    else:
        page["URL_of_Anchor"] = 0

    resources = (
        soup.find_all(
            ["img", "script", "link", "video", "audio", "source"],
            src=True
        )
        + soup.find_all(
            "link",
            href=True
        )
    )

    if resources:

        external = 0

        for tag in resources:

            href = (
                tag.get("src")
                or tag.get("href")
                or ""
            )

            if href.startswith(
                ("http://", "https://")
            ):

                resource_host = (
                    urlparse(href).hostname
                    or ""
                ).lower()

                if (
                    resource_host
                    and resource_host != host
                ):
                    external += 1

        ratio = external / max(
            1,
            len(resources)
        )

        page["Request_URL"] = (
            1 if ratio < 0.22
            else -1
        )

        page["Links_in_tags"] = (
            1 if ratio < 0.17
            else (0 if ratio < 0.81 else -1)
        )

    else:

        page["Request_URL"] = 1
        page["Links_in_tags"] = 1

    forms = soup.find_all("form")

    suspicious_sfh = False
    email_form = False

    for form in forms:

        action = (
            form.get("action")
            or ""
        ).strip().lower()

        if (
            action.startswith("mailto:")
            or "@" in action
        ):
            email_form = True

        if action in ("", "about:blank"):

            suspicious_sfh = True

        elif action.startswith(
            ("http://", "https://")
        ):

            action_host = (
                urlparse(action).hostname
                or ""
            ).lower()

            if (
                action_host
                and action_host != host
            ):
                suspicious_sfh = True

    page["SFH"] = (
        -1 if suspicious_sfh else 1
    )

    page["Submitting_to_email"] = (
        -1 if email_form else 1
    )

    page["Favicon"] = 1

    page["Iframe"] = (
        -1
        if soup.find("iframe")
        else 1
    )

    scripts = " ".join(
        str(x)
        for x in soup.find_all("script")
    )

    page["on_mouseover"] = (
        -1
        if re.search(
            r"onmouseover\s*=",
            html,
            re.I
        )
        else 1
    )

    page["RightClick"] = (
        -1
        if re.search(
            r"oncontextmenu\s*=|contextmenu",
            html,
            re.I
        )
        else 1
    )

    page["popUpWidnow"] = (
        -1
        if re.search(
            r"window\.open\s*\(",
            scripts,
            re.I
        )
        else 1
    )

    page["Redirect"] = (
        1
        if soup.find(
            "meta",
            attrs={
                "http-equiv":
                re.compile(
                    "refresh",
                    re.I
                )
            }
        )
        else 0
    )

    return page


def dns_features(host):

    try:
        socket.gethostbyname(host)
        dns = 1

    except Exception:
        dns = -1

    return {
        "DNSRecord": dns,
        "age_of_domain": 0,
        "web_traffic": 0,
        "Page_Rank": 0,
        "Google_Index": 0,
        "Links_pointing_to_page": 0,
        "Statistical_report": 1,
    }


def build_features(final_url, html):

    features = lexical_features(final_url)

    features.update(
        analyze_page(
            html,
            final_url
        )
    )

    host = (
        urlparse(final_url).hostname
        or ""
    )

    features.update(
        dns_features(host)
    )

    defaults = {
        "Domain_registeration_length": 0,
        "Favicon": 1,
        "port": 1,
        "HTTPS_token": 1,
        "Request_URL": 1,
        "URL_of_Anchor": 0,
        "Links_in_tags": 0,
        "SFH": 1,
        "Submitting_to_email": 1,
        "Abnormal_URL": 1,
        "Redirect": 0,
        "on_mouseover": 1,
        "RightClick": 1,
        "popUpWidnow": 1,
        "Iframe": 1,
        "age_of_domain": 0,
        "DNSRecord": 1,
        "web_traffic": 0,
        "Page_Rank": 0,
        "Google_Index": 0,
        "Links_pointing_to_page": 0,
        "Statistical_report": 1,
    }

    for key, value in defaults.items():

        features.setdefault(
            key,
            value
        )
X = np.array(
    [[features[column] for column in FEATURES]],
    dtype=float
)

return X, features


@app.get("/api/health")
def health():

    _, threshold = load_model()

    return {
        "status": "ok",
        "threshold": threshold
    }


@app.post("/api/scan")
def scan(payload: ScanRequest):

    try:

        model, threshold = load_model()

        final_url, html, status_code = safe_fetch(
            payload.url
        )

        X, features = build_features(
            final_url,
            html
        )

        score = float(
            model.predict_proba(X)[0, 1]
        )

        phishing = score >= threshold

        checks = {

            "IP address in URL":
                features.get(
                    "having_IP_Address"
                ) == -1,

            "Long URL":
                features.get(
                    "URL_Length"
                ) == -1,

            "@ symbol":
                features.get(
                    "having_At_Symbol"
                ) == -1,

            "URL shortening":
                features.get(
                    "Shortining_Service"
                ) == -1,

            "Hyphen in domain":
                features.get(
                    "Prefix_Suffix"
                ) == -1,

            "Multiple subdomains":
                features.get(
                    "having_Sub_Domain"
                ) == -1,

            "No HTTPS":
                features.get(
                    "SSLfinal_State"
                ) == -1,

            "External form handler":
                features.get(
                    "SFH"
                ) == -1,

            "External anchors":
                features.get(
                    "URL_of_Anchor"
                ) == -1,

            "Iframe detected":
                features.get(
                    "Iframe"
                ) == -1,

            "Mouse-over script":
                features.get(
                    "on_mouseover"
                ) == -1,

            "Right-click script":
                features.get(
                    "RightClick"
                ) == -1,

            "Popup script":
                features.get(
                    "popUpWidnow"
                ) == -1,
        }

        suspicious = [
            name
            for name, bad in checks.items()
            if bad
        ]

        return {

            "ok": True,

            "verdict":
                "PHISHING"
                if phishing
                else "LEGITIMATE",

            "label":
                "FAKE / SUSPICIOUS"
                if phishing
                else "ORIGINAL / LEGITIMATE",

            "score":
                round(
                    score * 100,
                    2
                ),

            "threshold":
                round(
                    threshold * 100,
                    2
                ),

            "final_url":
                final_url,

            "http_status":
                status_code,

            "suspicious_features":
                suspicious,

            "features_analyzed":
                len(FEATURES),

            "disclaimer":
                "ML research/demo result; not a guarantee of safety."
        }

    except requests.RequestException as exc:

        raise HTTPException(
            status_code=502,
            detail=f"Could not fetch the website: {exc}"
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}"
        )
