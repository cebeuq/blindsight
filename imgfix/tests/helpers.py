"""Shared fixtures: tiny images/PDFs built in-process, request bodies, media-part finder, a throwaway test CA."""
import base64
import datetime
import io
import ipaddress
import json
import os

from PIL import Image


def png(w=40, h=30, color=(255, 0, 0)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return buf.getvalue()


def noise_png(w=64, h=64):
    buf = io.BytesIO()
    Image.frombytes("RGB", (w, h), os.urandom(w * h * 3)).save(buf, "PNG")
    return buf.getvalue()


def jpeg_cmyk(w=64, h=48):
    buf = io.BytesIO()
    Image.new("CMYK", (w, h), (0, 255, 255, 0)).save(buf, "JPEG")
    return buf.getvalue()


def b64(raw):
    return base64.b64encode(raw).decode()


def image_block(raw, media_type="image/png", **extra):
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64(raw)}, **extra}


def pdf_block(raw, **extra):
    return {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64(raw)},
            **extra}


def text_pdf(pages):
    """Minimal valid PDF with one Helvetica text line per page (so pdftotext has something to extract)."""
    n = len(pages)
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            ("<< /Type /Pages /Kids [%s] /Count %d >>" % (" ".join(f"{4 + 2 * i} 0 R" for i in range(n)), n)).encode(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    for i, t in enumerate(pages):
        stream = f"BT /F1 18 Tf 72 720 Td ({t}) Tj ET".encode()
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
                    f"/Contents {5 + 2 * i} 0 R >>".encode())
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out, offs = bytearray(b"%PDF-1.4\n"), []
    for k, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % k + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for o in offs:
        out += b"%010d 00000 n \n" % o
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


def pillow_pdf(n=2):
    """Image-only n-page PDF written by Pillow (no text layer)."""
    frames = [Image.new("RGB", (200, 100), (i * 60, 0, 0)) for i in range(n)]
    buf = io.BytesIO()
    frames[0].save(buf, "PDF", save_all=True, append_images=frames[1:])
    return buf.getvalue()


def body(*messages, model="text-model", **extra):
    return {"model": model, "max_tokens": 1024, "messages": list(messages), **extra}


def user(*content):
    return {"role": "user", "content": list(content)}


def text(t):
    return {"type": "text", "text": t}


def dumps(obj):
    return json.dumps(obj).encode()


MEDIA_TYPES = ("image", "document", "image_url", "file", "input_image", "input_file", "computer_screenshot")
GEMINI_KEYS = ("inline_data", "inlineData", "file_data", "fileData")
OPAQUE = ("tool_use", "server_tool_use", "mcp_tool_use", "function_call")
OPAQUE_KEYS = ("arguments", "function_call", "functionCall")


def media_blocks(node, opaque=OPAQUE):
    """All media parts anywhere below node, in every request format (Anthropic, OpenAI Chat/Responses incl. untyped
    vLLM image_url parts, Ollama "images" lists, Gemini inline/file data). Model-written JSON (tool_use input,
    function_call, tool-call arguments) is not content."""
    found = []
    if isinstance(node, list):
        for x in node:
            found += media_blocks(x, opaque)
    elif isinstance(node, dict):
        if (node.get("type") in MEDIA_TYPES or any(k in node for k in GEMINI_KEYS)
                or ("type" not in node and "image_url" in node)):
            found.append(node)
        imgs = node.get("images")
        if isinstance(imgs, list) and imgs:
            found.append({"images": imgs})
        if node.get("type") not in opaque:
            for k, v in node.items():
                if k not in OPAQUE_KEYS:
                    found += media_blocks(v, opaque)
    return found


def data_url(raw, media_type="image/png"):
    return f"data:{media_type};base64,{b64(raw)}"


def part_texts(node, types=("text", "input_text")):
    """Every text / input_text part string below node, in order (model-written JSON skipped)."""
    out = []
    if isinstance(node, list):
        for x in node:
            out += part_texts(x, types)
    elif isinstance(node, dict) and node.get("type") not in OPAQUE:
        if node.get("type") in types:
            out.append(node["text"])
        for k, v in node.items():
            if k not in OPAQUE_KEYS:
                out += part_texts(v, types)
    return out


def make_ca(directory, names=("localhost",), ips=("127.0.0.1",)):
    """Throwaway CA + server certificate (cryptography ships with mitmproxy) -> (ca.pem, server cert, server key)."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    now = datetime.datetime.now(datetime.timezone.utc)

    def cert(subject, issuer, key, sign_key, ca, sans=None):
        b = (x509.CertificateBuilder().subject_name(subject).issuer_name(issuer).public_key(key.public_key())
             .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
             .not_valid_after(now + datetime.timedelta(days=2))
             .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True))
        if sans:
            b = b.add_extension(x509.SubjectAlternativeName(sans), critical=False)
        return b.sign(sign_key, hashes.SHA256())

    ca_key, srv_key = ec.generate_private_key(ec.SECP256R1()), ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "blindsight test CA")])
    ca = cert(ca_name, ca_name, ca_key, ca_key, True)
    sans = [x509.DNSName(n) for n in names] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    srv = cert(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, names[0])]), ca_name, srv_key, ca_key, False, sans)
    pem = serialization.Encoding.PEM
    paths = [os.path.join(directory, n) for n in ("ca.pem", "server.pem", "server.key")]
    blobs = (ca.public_bytes(pem), srv.public_bytes(pem),
             srv_key.private_bytes(pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    for path, data in zip(paths, blobs):
        with open(path, "wb") as f:
            f.write(data)
    return paths


def texts(obj):
    """All text-block strings in the messages of a parsed body, in order."""
    out = []

    def walk(n):
        if isinstance(n, list):
            for x in n:
                walk(x)
        elif isinstance(n, dict):
            if n.get("type") == "text":
                out.append(n["text"])
            for k, v in n.items():
                if k != "input":
                    walk(v)
    walk(obj["messages"])
    return out
