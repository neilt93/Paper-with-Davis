#!/usr/bin/env python3
"""Simple verification app for the visibility dataset with inline editing."""

import csv
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import urllib.parse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Sheets", "results.csv")
VERIFIED_PATH = os.path.join(BASE_DIR, "Sheets", "verified.json")

FIELDNAMES = ['ID', 'Status', 'base_setup', 'base_question', 'flip_question', 'flip_change', 'pic_base', 'pic_flip', 'indoor_outdoor']

def load_csv():
    rows = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def save_csv(rows):
    with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in FIELDNAMES})

def load_verified():
    if os.path.exists(VERIFIED_PATH):
        with open(VERIFIED_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_verified(data):
    with open(VERIFIED_PATH, 'w') as f:
        json.dump(data, f, indent=2)

class Handler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress logs

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path == '/api/rows':
            rows = load_csv()
            verified = load_verified()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"rows": rows, "verified": verified}).encode())
        elif self.path.startswith('/image/'):
            img_path = urllib.parse.unquote(self.path[7:])
            img_path = img_path.strip().strip("'").strip('"')
            if os.path.exists(img_path):
                self.send_response(200)
                ext = os.path.splitext(img_path)[1].lower()
                ctype = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}.get(ext, 'image/jpeg')
                self.send_header('Content-type', ctype)
                self.end_headers()
                with open(img_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(f"Not found: {img_path}".encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode()
        data = json.loads(body)

        if self.path == '/api/verify':
            verified = load_verified()
            verified[data['id']] = data['status']
            save_verified(verified)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode())
        elif self.path == '/api/update':
            rows = load_csv()
            for row in rows:
                if row['ID'] == data['ID']:
                    for k, v in data.items():
                        row[k] = v
                    break
            save_csv(rows)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode())
        else:
            self.send_response(404)
            self.end_headers()

HTML = '''<!DOCTYPE html>
<html>
<head>
    <title>Verify Dataset</title>
    <style>
        body { font-family: system-ui; max-width: 1400px; margin: 0 auto; padding: 20px; background: #1a1a1a; color: #fff; }
        .row { border: 2px solid #444; padding: 20px; margin: 20px 0; border-radius: 8px; background: #2a2a2a; }
        .row.correct { border-color: #4a4; background: #1a2a1a; }
        .row.wrong { border-color: #a44; background: #2a1a1a; }
        .row.edited { border-color: #aa4; }
        .images { display: flex; gap: 20px; margin: 15px 0; }
        .img-box { flex: 1; text-align: center; }
        .img-box img { max-width: 100%; max-height: 350px; border-radius: 4px; cursor: pointer; }
        .img-box img:hover { opacity: 0.8; }
        .img-box p { margin: 5px 0; font-size: 12px; color: #aaa; }
        h2 { margin: 0 0 10px 0; color: #fff; display: flex; align-items: center; gap: 10px; }
        h2 span { font-size: 12px; color: #888; font-weight: normal; }
        .field { margin: 10px 0; }
        .field label { display: block; font-size: 12px; color: #888; margin-bottom: 4px; }
        .field input, .field textarea { width: 100%; padding: 8px; border: 1px solid #555; border-radius: 4px; background: #333; color: #fff; font-size: 14px; box-sizing: border-box; }
        .field textarea { min-height: 60px; resize: vertical; }
        .field input:focus, .field textarea:focus { border-color: #88f; outline: none; }
        .buttons { display: flex; gap: 10px; margin-top: 15px; flex-wrap: wrap; }
        button { padding: 10px 20px; font-size: 14px; border: none; border-radius: 4px; cursor: pointer; }
        .btn-correct { background: #4a4; color: white; }
        .btn-wrong { background: #a44; color: white; }
        .btn-save { background: #48f; color: white; }
        .btn-clear { background: #666; color: white; }
        .stats { margin-bottom: 10px; font-size: 14px; background: #333; padding: 10px; border-radius: 4px; }
        .filter { margin-bottom: 20px; }
        .filter button { margin-right: 10px; padding: 8px 16px; background: #444; color: #fff; }
        .filter button.active { background: #558; }
        .saved { color: #4a4; font-size: 12px; margin-left: 10px; }
        .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    </style>
</head>
<body>
    <h1>Dataset Verification</h1>
    <div class="filter">
        <button onclick="setFilter('all')" id="f-all">All</button>
        <button onclick="setFilter('unverified')" id="f-unverified">Unverified</button>
        <button onclick="setFilter('correct')" id="f-correct">Correct</button>
        <button onclick="setFilter('wrong')" id="f-wrong">Wrong</button>
    </div>
    <div class="stats" id="stats"></div>
    <div id="content"></div>

    <script>
        let rows = [];
        let verified = {};
        let filter = 'all';
        let edited = {};

        async function load() {
            const res = await fetch('/api/rows');
            const data = await res.json();
            rows = data.rows;
            verified = data.verified;
            render();
        }

        function setFilter(f) {
            filter = f;
            document.querySelectorAll('.filter button').forEach(b => b.classList.remove('active'));
            document.getElementById('f-' + f).classList.add('active');
            render();
        }

        function cleanPath(p) {
            return (p || '').trim().replace(/^'+|'+$/g, '').replace(/^"+|"+$/g, '');
        }

        function render() {
            const total = rows.length;
            const correct = Object.values(verified).filter(v => v === 'correct').length;
            const wrong = Object.values(verified).filter(v => v === 'wrong').length;
            document.getElementById('stats').innerHTML =
                `Total: ${total} | Verified: ${correct + wrong} | Correct: <span style="color:#4a4">${correct}</span> | Wrong: <span style="color:#a44">${wrong}</span> | Remaining: ${total - correct - wrong}`;

            let filtered = rows;
            if (filter === 'unverified') filtered = rows.filter(r => !verified[r.ID]);
            else if (filter === 'correct') filtered = rows.filter(r => verified[r.ID] === 'correct');
            else if (filter === 'wrong') filtered = rows.filter(r => verified[r.ID] === 'wrong');

            const html = filtered.map((row, idx) => {
                const status = verified[row.ID] || '';
                const isEdited = edited[row.ID] ? 'edited' : '';
                const picBase = cleanPath(row.pic_base);
                const picFlip = cleanPath(row.pic_flip);
                return `
                <div class="row ${status} ${isEdited}" id="row-${row.ID}">
                    <h2>${row.ID} <span>${row.indoor_outdoor || ''}</span></h2>

                    <div class="field">
                        <label>Base Setup</label>
                        <input type="text" value="${esc(row.base_setup)}" onchange="edit('${row.ID}', 'base_setup', this.value)">
                    </div>

                    <div class="images">
                        <div class="img-box">
                            <img src="/image/${encodeURIComponent(picBase)}" alt="base" onclick="window.open('/image/${encodeURIComponent(picBase)}')">
                            <p>BASE</p>
                            <input type="text" value="${esc(row.pic_base)}" onchange="edit('${row.ID}', 'pic_base', this.value); reloadImg(this)" style="font-size:11px;">
                        </div>
                        <div class="img-box">
                            <img src="/image/${encodeURIComponent(picFlip)}" alt="flip" onclick="window.open('/image/${encodeURIComponent(picFlip)}')">
                            <p>FLIP</p>
                            <input type="text" value="${esc(row.pic_flip)}" onchange="edit('${row.ID}', 'pic_flip', this.value); reloadImg(this)" style="font-size:11px;">
                        </div>
                    </div>

                    <div class="two-col">
                        <div class="field">
                            <label>Base Question</label>
                            <textarea onchange="edit('${row.ID}', 'base_question', this.value)">${esc(row.base_question)}</textarea>
                        </div>
                        <div class="field">
                            <label>Flip Question</label>
                            <textarea onchange="edit('${row.ID}', 'flip_question', this.value)">${esc(row.flip_question)}</textarea>
                        </div>
                    </div>

                    <div class="field">
                        <label>Flip Change (what changes between base and flip image)</label>
                        <input type="text" value="${esc(row.flip_change)}" onchange="edit('${row.ID}', 'flip_change', this.value)">
                    </div>

                    <div class="buttons">
                        <button class="btn-save" onclick="saveRow('${row.ID}')">Save Changes</button>
                        <button class="btn-correct" onclick="verify('${row.ID}', 'correct')">Correct</button>
                        <button class="btn-wrong" onclick="verify('${row.ID}', 'wrong')">Wrong</button>
                        <button class="btn-clear" onclick="verify('${row.ID}', '')">Clear Status</button>
                        <span class="saved" id="saved-${row.ID}"></span>
                    </div>
                </div>`;
            }).join('');
            document.getElementById('content').innerHTML = html;
        }

        function esc(s) {
            return (s || '').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        }

        function edit(id, field, value) {
            const row = rows.find(r => r.ID === id);
            if (row) {
                row[field] = value;
                edited[id] = true;
                document.getElementById('row-' + id)?.classList.add('edited');
            }
        }

        function reloadImg(input) {
            const img = input.parentElement.querySelector('img');
            const path = cleanPath(input.value);
            img.src = '/image/' + encodeURIComponent(path);
        }

        async function saveRow(id) {
            const row = rows.find(r => r.ID === id);
            if (!row) return;
            await fetch('/api/update', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(row)
            });
            delete edited[id];
            document.getElementById('row-' + id)?.classList.remove('edited');
            document.getElementById('saved-' + id).textContent = 'Saved!';
            setTimeout(() => {
                const el = document.getElementById('saved-' + id);
                if (el) el.textContent = '';
            }, 2000);
        }

        async function verify(id, status) {
            await fetch('/api/verify', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({id, status})
            });
            verified[id] = status;
            const el = document.getElementById('row-' + id);
            if (el) {
                el.classList.remove('correct', 'wrong');
                if (status) el.classList.add(status);
            }
            // Update stats
            const total = rows.length;
            const correct = Object.values(verified).filter(v => v === 'correct').length;
            const wrong = Object.values(verified).filter(v => v === 'wrong').length;
            document.getElementById('stats').innerHTML =
                `Total: ${total} | Verified: ${correct + wrong} | Correct: <span style="color:#4a4">${correct}</span> | Wrong: <span style="color:#a44">${wrong}</span> | Remaining: ${total - correct - wrong}`;
        }

        load();
        setFilter('all');
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    port = 8765
    print(f"Starting server at http://localhost:{port}")
    print(f"CSV: {CSV_PATH}")
    HTTPServer(('localhost', port), Handler).serve_forever()
