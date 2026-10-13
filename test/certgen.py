"""Test-only TLS material for the stack test (test/run.sh): a throwaway CA and one server certificate signed by it.

    python3 -I certgen.py OUTDIR NAME [NAME ...]
writes OUTDIR/ca.pem (the CA certificate: what TEXT_CA_FILE points at), OUTDIR/server.pem and OUTDIR/server.key
(for the stub's TLS listener), and OUTDIR/other-ca.pem: an unrelated CA that did not sign the server certificate
(TEXT_CA_FILE pointing at it must make the router refuse the server). NAMEs become subjectAltNames (IP addresses
as IP entries). Valid for 2 days; the key is world-readable on purpose (the stub runs as nobody): never use any of
this outside a test. Needs only the cryptography package (in the imgfix image: mitmproxy depends on it).
"""
import datetime
import ipaddress
import os
import sys

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


def name(cn):
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def make_ca(cn, start, end):
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca = (x509.CertificateBuilder().subject_name(name(cn)).issuer_name(name(cn))
          .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
          .not_valid_before(start).not_valid_after(end)
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.KeyUsage(digital_signature=False, content_commitment=False, key_encipherment=False,
                                       data_encipherment=False, key_agreement=False, key_cert_sign=True,
                                       crl_sign=True, encipher_only=False, decipher_only=False), critical=True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
          .sign(ca_key, hashes.SHA256()))
    return ca, ca_key


def main(out, names):
    now = datetime.datetime.now(datetime.timezone.utc)
    start, end = now - datetime.timedelta(hours=1), now + datetime.timedelta(days=2)
    ca, ca_key = make_ca("blindsight test CA", start, end)
    other, _ = make_ca("blindsight test CA", start, end)   # same name, different key: still untrusted
    key = ec.generate_private_key(ec.SECP256R1())
    sans = []
    for n in names:
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(n)))
        except ValueError:
            sans.append(x509.DNSName(n))
    cert = (x509.CertificateBuilder().subject_name(name(names[0])).issuer_name(ca.subject)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(start).not_valid_after(end)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    pem = serialization.Encoding.PEM
    files = {"ca.pem": ca.public_bytes(pem), "other-ca.pem": other.public_bytes(pem),
             "server.pem": cert.public_bytes(pem),
             "server.key": key.private_bytes(pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())}
    for fn, data in files.items():
        path = os.path.join(out, fn)
        with open(path, "wb") as f:
            f.write(data)
        os.chmod(path, 0o644)
    print(f"certgen: {', '.join(files)} in {out} for {', '.join(names)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
