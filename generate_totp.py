#!/usr/bin/env python3

import argparse
import base64
import hashlib
import hmac
import struct
import time


def generate_totp(base32_secret: str, digits: int = 6, period: int = 30) -> str:
    # Normalize common user input formats (spaces, lowercase, missing padding).
    normalized = base32_secret.strip().replace(" ", "").upper()
    padding = (-len(normalized)) % 8
    normalized += "=" * padding

    key = base64.b32decode(normalized, casefold=True)
    counter = int(time.time() // period)
    msg = struct.pack(">Q", counter)
    digest = hmac.new(key, msg, hashlib.sha1).digest()

    offset = digest[-1] & 0x0F
    code_int = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
    code = code_int % (10**digits)
    return str(code).zfill(digits)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the current 6-digit TOTP code from a base32 secret."
    )
    parser.add_argument("secret", help="Base32-encoded TOTP secret")
    args = parser.parse_args()

    print(generate_totp(args.secret))


if __name__ == "__main__":
    main()
