"""Phase 7 local demo: upload a PRE/POST pair (or pick a validation example) and get a damage map.

python -m src.demo_app            # then open http://127.0.0.1:7860

No extra packages: Python's standard-library HTTP server + the Phase 5 two-stage pipeline with the
settings frozen in the Phase 6 protocol (U-Net threshold 0.9, split erosion 4; ResNet34 early fusion).
Listens on 127.0.0.1 only. Examples are validation images (never the test split).
"""
import argparse
import base64
import io
import json
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

from src.data.dataset import ROOT
from src.evaluation.two_stage import (CLASS_NAMES, DAMAGE_COLORS, DEFAULT_CLASSIFIER, DEFAULT_UNET, MIN_AREA,
                                      draw_damage, load_models, run_two_stage)

THRESHOLD, SPLIT_EROSION = 0.9, 4
MAX_BODY_BYTES = 40 * 1024 * 1024
EXAMPLES = {
    "palu-tsunami_00000025": "Palu tsunamisi — sahilde yıkılmış binalar",
    "santa-rosa-wildfire_00000007": "Santa Rosa yangını — yanmış ve sağlam evler",
    "hurricane-michael_00000180": "Michael kasırgası — dört hasar düzeyi",
    "hurricane-harvey_00000434": "Harvey kasırgası (sel) — ağır hasar",
    "mexico-earthquake_00000119": "Meksika depremi — çoğunlukla sağlam",
}
EXAMPLE_DIR = "data/train/images"
LABELS_TR = {"no-damage": "Hasarsız", "minor-damage": "Hafif hasar",
             "major-damage": "Ağır hasar", "destroyed": "Yıkılmış"}


class DemoError(ValueError):
    """A problem with the user's input, shown on the page."""


def decode_image(data_url):
    if not isinstance(data_url, str) or "," not in data_url:
        raise DemoError("Görüntü okunamadı")
    try:
        image = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1], validate=True)))
        image.load()
    except Exception as exc:
        raise DemoError("Görüntü okunamadı (PNG veya JPG yükleyin)") from exc
    return image.convert("RGB")


def check_pair(pre, post):
    if pre.size != post.size:
        raise DemoError(f"PRE ve POST aynı boyutta olmalı ({pre.size[0]}×{pre.size[1]} ≠ {post.size[0]}×{post.size[1]})")
    width, height = pre.size
    if width % 32 or height % 32 or not 256 <= min(width, height) <= max(width, height) <= 2048:
        raise DemoError("Görüntü kenarları 32'nin katı ve 256–2048 piksel olmalı (xBD: 1024×1024)")


def missing_checkpoints(project_root=ROOT):
    """Checkpoint paths (relative to the project) that are not on disk."""
    return [path for path in (DEFAULT_UNET, DEFAULT_CLASSIFIER) if not (Path(project_root) / path).is_file()]


def example_files(image_id, project_root=ROOT):
    base = Path(project_root) / EXAMPLE_DIR
    return base / f"{image_id}_pre_disaster.png", base / f"{image_id}_post_disaster.png"


def available_examples(project_root=ROOT):
    """Examples whose original PRE and POST images are both on disk."""
    return {k: v for k, v in EXAMPLES.items() if all(f.is_file() for f in example_files(k, project_root))}


def example_pair(image_id, project_root=ROOT):
    if image_id not in EXAMPLES:
        raise DemoError("Bilinmeyen örnek")
    pre_path, post_path = example_files(image_id, project_root)
    missing = [f.name for f in (pre_path, post_path) if not f.is_file()]
    if missing:
        raise DemoError(f"Örnek görüntüsü bulunamadı: {EXAMPLE_DIR}/{', '.join(missing)} "
                        "(xBD train görüntüleri gerekir; kendi PRE/POST görüntülerinizi yükleyebilirsiniz)")
    return Image.open(pre_path).convert("RGB"), Image.open(post_path).convert("RGB")


def to_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def predict_pair(pre, post, unet, classifier, device):
    check_pair(pre, post)
    started = time.perf_counter()
    labels, detections = run_two_stage(pre, post, unet, classifier, device, threshold=THRESHOLD,
                                       min_area=MIN_AREA, split_erosion=SPLIT_EROSION)
    counts = Counter(d["predicted_damage"] for d in detections)
    return dict(overlay=to_data_url(draw_damage(post, labels, detections)), pre=to_data_url(pre),
                buildings=len(detections), seconds=round(time.perf_counter() - started, 2),
                counts=[dict(name=LABELS_TR[n], key=n, count=counts.get(n, 0),
                             color="rgb({},{},{})".format(*DAMAGE_COLORS[i])) for i, n in enumerate(CLASS_NAMES)])


def page_html(project_root=ROOT):
    examples = available_examples(project_root)
    options = "".join(f'<option value="{k}">{v}</option>' for k, v in examples.items())
    note = "" if examples else (f"Hazır örneklerin görüntüleri bulunamadı ({EXAMPLE_DIR}). "
                                "Kendi PRE/POST görüntülerinizi yükleyebilirsiniz.")
    return PAGE.replace("{{OPTIONS}}", options).replace("{{NOTE}}", note).encode("utf-8")


def make_handler(unet, classifier, device, project_root=ROOT):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the terminal quiet
            pass

        def _send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, page_html(project_root), "text/html; charset=utf-8")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/predict":
                return self._send(404, b"not found", "text/plain")
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY_BYTES:
                return self._json(413, dict(error="İstek çok büyük (en çok 40 MB)"))
            try:
                request = json.loads(self.rfile.read(length))
                if request.get("example"):
                    pre, post = example_pair(request["example"], project_root)
                else:
                    pre, post = decode_image(request.get("pre")), decode_image(request.get("post"))
                self._json(200, predict_pair(pre, post, unet, classifier, device))
            except DemoError as exc:
                self._json(400, dict(error=str(exc)))
            except (json.JSONDecodeError, AttributeError):
                self._json(400, dict(error="Geçersiz istek"))
            except Exception as exc:  # unexpected: keep the page usable and show the reason
                self._json(500, dict(error=f"Beklenmeyen hata: {exc}"))

        def _json(self, status, payload):
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    return Handler


def main(argv=None):
    import torch
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    missing = missing_checkpoints()
    if missing:
        raise SystemExit("Demo başlatılamadı: model ağırlıkları bulunamadı:\n"
                         + "\n".join(f"  - {ROOT / path}" for path in missing)
                         + "\nBu dosyalar git'te yok; Faz 4/5 eğitim klasörlerinden ya da yedekten kopyalayın.")
    if not available_examples():
        print(f"Uyarı: hazır örnek görüntüleri bulunamadı ({ROOT / EXAMPLE_DIR}); yalnızca yükleme çalışır.", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Modeller yükleniyor ({device}) ...", flush=True)
    unet, classifier, meta = load_models(device)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(unet, classifier, device))
    print(f"Demo hazır: http://127.0.0.1:{args.port}  (durdurmak için Ctrl+C)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


PAGE = """<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hasar Tespiti Demo</title>
<style>
:root{--bg:#F6F4EF;--card:#FFFDF8;--ink:#14213D;--muted:#4A5568;--line:#DDD6C8;--accent:#B4531E}
@media (prefers-color-scheme:dark){:root{--bg:#10192E;--card:#18243F;--ink:#F1EEE7;--muted:#B8C2D6;--line:#2B3A5C;--accent:#E8884E}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,Segoe UI,Arial,sans-serif}
main{max-width:1200px;margin:0 auto;padding:32px 16px}h1{font-size:28px;margin:0 0 4px}p.lead{color:var(--muted);margin:0 0 24px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin-bottom:20px}
.row{display:flex;flex-wrap:wrap;gap:16px;align-items:end}label{display:flex;flex-direction:column;gap:6px;font-weight:600;font-size:14px}
select,input{font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px;max-width:100%}
button{font:inherit;font-weight:600;background:var(--accent);color:#fff;border:0;border-radius:8px;padding:10px 18px;cursor:pointer}
button:disabled{opacity:.6;cursor:wait}.or{color:var(--muted);align-self:center}
.images{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}figure{margin:0}
figure img{width:100%;border-radius:8px;border:1px solid var(--line);display:block}figcaption{font-size:14px;color:var(--muted);margin-top:6px}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin-top:12px}.chip{display:flex;align-items:center;gap:8px;font-size:14px}
.sw{width:14px;height:14px;border-radius:3px;display:inline-block}#status{color:var(--muted);min-height:24px}.err{color:#C0392B}
</style></head><body><main>
<h1>Afet sonrası bina hasarı — demo</h1>
<p class="lead">U-Net afet öncesi görüntüde binaları bulur, sınıflandırıcı her binanın hasarını öncesi/sonrası çiftinden tahmin eder.</p>
<div class="card"><div class="row">
<label>Hazır örnek (validation)<select id="example"><option value="">— seç —</option>{{OPTIONS}}</select></label>
<span class="or">veya</span>
<label>PRE (afet öncesi)<input type="file" id="pre" accept="image/png,image/jpeg"></label>
<label>POST (afet sonrası)<input type="file" id="post" accept="image/png,image/jpeg"></label>
<button id="run">Hasar haritası çıkar</button></div>
<p id="status" role="status">{{NOTE}}</p></div>
<div class="card" id="result" hidden>
<div class="images"><figure><img id="img-pre" alt="Afet öncesi görüntü"><figcaption>PRE — afet öncesi</figcaption></figure>
<figure><img id="img-out" alt="Afet sonrası görüntü üzerinde tahmin edilen hasar haritası"><figcaption id="summary"></figcaption></figure></div>
<div class="legend" id="legend"></div></div>
<p style="color:var(--muted);font-size:13px">Ayarlar Faz 6 protokolündekiyle aynı: eşik 0,9, bina ayırma 4 · Sınıflandırıcı: ResNet34 early fusion · Yalnızca bu bilgisayarda çalışır.</p>
</main><script>
const $ = id => document.getElementById(id);
const read = f => new Promise((ok, no) => { const r = new FileReader(); r.onload = () => ok(r.result); r.onerror = no; r.readAsDataURL(f); });
$('run').onclick = async () => {
  const example = $('example').value, pre = $('pre').files[0], post = $('post').files[0];
  if (!example && !(pre && post)) { $('status').textContent = 'Bir örnek seçin ya da PRE ve POST görüntüsünü yükleyin.'; return; }
  $('run').disabled = true; $('status').className = ''; $('status').textContent = 'Çalışıyor…';
  try {
    const body = example ? {example} : {pre: await read(pre), post: await read(post)};
    const res = await fetch('/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Hata');
    $('img-pre').src = data.pre; $('img-out').src = data.overlay;
    $('summary').textContent = `POST + tahmin — ${data.buildings} bina adayı, ${data.seconds} sn`;
    $('legend').innerHTML = data.counts.map(c => `<span class="chip"><span class="sw" style="background:${c.color}"></span>${c.name}: <b>${c.count}</b></span>`).join('');
    $('result').hidden = false; $('status').textContent = '';
  } catch (e) { $('status').className = 'err'; $('status').textContent = e.message; }
  finally { $('run').disabled = false; }
};
</script></body></html>"""


if __name__ == "__main__":
    main()
