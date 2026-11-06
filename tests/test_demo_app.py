"""Phase 7 demo server checks with stub models (CPU, no xBD data, no downloads)."""
import base64
import io
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
import torch
from PIL import Image

from src import demo_app
from src.demo_app import (DemoError, available_examples, check_pair, decode_image, example_pair, make_handler,
                          missing_checkpoints, page_html, predict_pair, to_data_url)
from test_two_stage import BrightnessClassifier, FixedUNet


def pair(size=256):
    pre = np.full((size, size, 3), 200, dtype=np.uint8)
    post = pre.copy()
    mask = np.zeros((size, size), dtype=bool)
    mask[20:60, 20:60] = True            # intact building
    mask[150:200, 150:210] = True        # destroyed building: dark in POST
    post[140:210, 140:220] = 20
    return Image.fromarray(pre), Image.fromarray(post), mask


def test_decode_image_roundtrip_and_errors():
    image = Image.new("RGB", (64, 32), (10, 20, 30))
    decoded = decode_image(to_data_url(image))
    assert decoded.size == (64, 32) and decoded.getpixel((0, 0)) == (10, 20, 30)
    for bad in (None, "no-comma", "data:image/png;base64,!!!", "data:image/png;base64," + base64.b64encode(b"x").decode()):
        with pytest.raises(DemoError):
            decode_image(bad)


def test_check_pair_rules():
    check_pair(Image.new("RGB", (1024, 1024)), Image.new("RGB", (1024, 1024)))
    with pytest.raises(DemoError, match="aynı boyutta"):
        check_pair(Image.new("RGB", (256, 256)), Image.new("RGB", (512, 512)))
    with pytest.raises(DemoError, match="32'nin katı"):
        check_pair(Image.new("RGB", (300, 300)), Image.new("RGB", (300, 300)))
    with pytest.raises(DemoError, match="32'nin katı"):
        check_pair(Image.new("RGB", (128, 128)), Image.new("RGB", (128, 128)))


def test_predict_pair_counts_and_overlay():
    pre, post, mask = pair()
    result = predict_pair(pre, post, FixedUNet(mask), BrightnessClassifier(), torch.device("cpu"))
    assert result["buildings"] == 2
    counts = {c["key"]: c["count"] for c in result["counts"]}
    assert counts == {"no-damage": 1, "minor-damage": 0, "major-damage": 0, "destroyed": 1}
    overlay = decode_image(result["overlay"])
    assert overlay.size == (256, 256) and overlay.getpixel((40, 40)) != post.getpixel((40, 40))


@pytest.fixture
def server(tmp_path):
    pre, post, mask = pair()
    images = tmp_path / demo_app.EXAMPLE_DIR
    images.mkdir(parents=True)
    pre.save(images / "palu-tsunami_00000025_pre_disaster.png")
    post.save(images / "palu-tsunami_00000025_post_disaster.png")
    handler = make_handler(FixedUNet(mask), BrightnessClassifier(), torch.device("cpu"), project_root=tmp_path)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", pre, post
    httpd.shutdown()
    httpd.server_close()


def post_json(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_http_page_and_predict(server):
    base, pre, post = server
    with urllib.request.urlopen(base + "/", timeout=30) as response:
        page = response.read().decode("utf-8")
    assert response.status == 200 and "Hasar haritası çıkar" in page and "palu-tsunami_00000025" in page
    status, body = post_json(base + "/predict", {"pre": to_data_url(pre), "post": to_data_url(post)})
    assert status == 200 and body["buildings"] == 2 and body["overlay"].startswith("data:image/png;base64,")
    status, body = post_json(base + "/predict", {"pre": to_data_url(pre), "post": to_data_url(post.resize((128, 128)))})
    assert status == 400 and "aynı boyutta" in body["error"]
    status, body = post_json(base + "/predict", {"example": "palu-tsunami_00000025"})
    assert status == 200 and body["buildings"] == 2
    status, body = post_json(base + "/predict", {"example": "santa-rosa-wildfire_00000007"})
    assert status == 400 and "bulunamadı" in body["error"]
    status, body = post_json(base + "/predict", {"example": "unknown_00000001"})
    assert status == 400 and "Bilinmeyen" in body["error"]
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(base + "/missing", timeout=30)


def test_missing_files_are_reported(tmp_path):
    assert set(missing_checkpoints(tmp_path)) == {demo_app.DEFAULT_UNET, demo_app.DEFAULT_CLASSIFIER}
    for path in missing_checkpoints(tmp_path):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes(b"x")
    assert missing_checkpoints(tmp_path) == []

    assert available_examples(tmp_path) == {}
    page = page_html(tmp_path).decode("utf-8")
    assert "palu-tsunami_00000025" not in page and "bulunamadı" in page
    with pytest.raises(DemoError, match="bulunamadı"):
        example_pair("palu-tsunami_00000025", tmp_path)

    images = tmp_path / demo_app.EXAMPLE_DIR
    images.mkdir(parents=True)
    for kind in ("pre", "post"):
        Image.new("RGB", (256, 256)).save(images / f"palu-tsunami_00000025_{kind}_disaster.png")
    assert list(available_examples(tmp_path)) == ["palu-tsunami_00000025"]
    pre, post = example_pair("palu-tsunami_00000025", tmp_path)
    assert pre.size == post.size == (256, 256)
    page = page_html(tmp_path).decode("utf-8")
    assert "palu-tsunami_00000025" in page and "santa-rosa-wildfire_00000007" not in page


def test_main_stops_with_clear_message_without_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(demo_app, "ROOT", tmp_path)
    monkeypatch.setattr(demo_app, "missing_checkpoints", lambda root=tmp_path: [demo_app.DEFAULT_UNET])
    with pytest.raises(SystemExit, match="model ağırlıkları bulunamadı"):
        demo_app.main([])
