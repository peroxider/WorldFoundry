import time

from worldfoundry_inf.security import HmacTokenAuth


def test_sign_and_verify_roundtrip():
    auth = HmacTokenAuth(secret="top-secret")
    body = b'{"video":"abc"}'
    ts = str(int(time.time()))
    sig = auth.sign("POST", "/v1/world/encode", body, ts)
    assert auth.verify("POST", "/v1/world/encode", body, ts, sig)


def test_verify_rejects_tampered_body():
    auth = HmacTokenAuth(secret="top-secret")
    ts = str(int(time.time()))
    sig = auth.sign("POST", "/v1/world/encode", b"original", ts)
    assert not auth.verify("POST", "/v1/world/encode", b"tampered", ts, sig)


def test_verify_rejects_stale_timestamp():
    auth = HmacTokenAuth(secret="top-secret", max_skew_seconds=10)
    sig = auth.sign("POST", "/v1/world/encode", b"", "1000000000")  # year 2001
    assert not auth.verify("POST", "/v1/world/encode", b"", "1000000000", sig)
