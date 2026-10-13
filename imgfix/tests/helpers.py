"""Shared fixtures: tiny images/PDFs built in-process, Anthropic-shaped bodies, media-block finder."""
import base64
import io
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


def media_blocks(node, opaque=("tool_use", "server_tool_use", "mcp_tool_use")):
    """All image/document blocks anywhere below node (tool_use.input is model JSON, not content)."""
    found = []
    if isinstance(node, list):
        for x in node:
            found += media_blocks(x, opaque)
    elif isinstance(node, dict):
        if node.get("type") in ("image", "document"):
            found.append(node)
        if node.get("type") not in opaque:
            for v in node.values():
                found += media_blocks(v, opaque)
    return found


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
