
from pathlib import Path
from urllib.parse import urlsplit
import ipaddress
import re

import joblib
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "phishing_model.joblib"

app = Flask(__name__)

model = joblib.load(MODEL_PATH)

SUSPICIOUS_WORDS = {
    "login", "verify", "update", "secure",
    "account", "password", "confirm", "free",
    "prize", "bank", "wallet", "signin",
}


def analyze_url(url):
    url = url.strip()

    if not url or len(url) > 2048:
        return {"error": "Enter a URL shorter than 2049 characters."}

    candidate = url if "://" in url else "https://" + url

    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname

        # Only accept normal web URLs.
        if (
            parsed.scheme not in ("http", "https")
            or not hostname
            or any(char.isspace() for char in candidate)
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("Invalid URL")

        # Basic hostname validation.
        hostname = hostname.encode("idna").decode("ascii").lower()
        if "." not in hostname and hostname != "localhost":
            raise ValueError("Invalid hostname")

    except (ValueError, UnicodeError):
        return {
            "error": "Enter a valid HTTP or HTTPS URL, such as https://example.com."
        }

    text_to_check = candidate.lower()
    flags = []

    if parsed.scheme != "https":
        flags.append("The URL does not use HTTPS.")

    try:
        ipaddress.ip_address(hostname.strip("[]"))
        flags.append("The hostname is an IP address.")
    except ValueError:
        pass

    if len(hostname) > 50:
        flags.append("The hostname is unusually long.")

    if hostname.count(".") >= 4:
        flags.append("The hostname has many subdomains.")

    if "@" in parsed.netloc:
        flags.append("The URL contains an @ symbol in its authority.")

    matched_words = sorted(
        word for word in SUSPICIOUS_WORDS if word in text_to_check
    )
    if matched_words:
        flags.append(
            "Potentially suspicious words: " + ", ".join(matched_words)
        )

    try:
        prediction = int(model.predict([candidate])[0])
        probabilities = model.predict_proba([candidate])[0]
        phishing_index = list(model.classes_).index(1)
        model_score = round(float(probabilities[phishing_index]) * 100, 1)
    except (ValueError, AttributeError, IndexError):
        prediction = 0
        model_score = 0.0
        flags.append("The model could not provide a reliable prediction.")

    # This is a heuristic indicator, not a validated risk probability.
    if prediction == 1 or len(flags) >= 2:
        result = "Potentially suspicious"
    else:
        result = "No obvious warning signs detected"

    return {
        "url": candidate,
        "result": result,
        "model_score": model_score,
        "flags": flags,
    }


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    url = ""

    if request.method == "POST":
        url = request.form.get("url", "")
        result = analyze_url(url)

    return render_template("index.html", result=result, url=url)


if __name__ == "__main__":
    app.run(debug=True)
