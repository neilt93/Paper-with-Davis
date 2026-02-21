#!/usr/bin/env python3
"""Image pair to row matching tool with undo support."""

import csv
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import urllib.parse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Sheets", "results.csv")
VERIFIED_PATH = os.path.join(BASE_DIR, "Sheets", "verified.json")
MATCHED_PATH = os.path.join(BASE_DIR, "Sheets", "matched.json")  # Track matches

FIELDNAMES = ['ID', 'Status', 'base_setup', 'base_question', 'flip_question', 'flip_change', 'pic_base', 'pic_flip', 'indoor_outdoor']

def load_verified():
    if os.path.exists(VERIFIED_PATH):
        with open(VERIFIED_PATH, 'r') as f:
            return json.load(f)
    return {}

def load_matched():
    if os.path.exists(MATCHED_PATH):
        with open(MATCHED_PATH, 'r') as f:
            return json.load(f)
    return {"matches": [], "original": {}}

def save_matched(data):
    with open(MATCHED_PATH, 'w') as f:
        json.dump(data, f, indent=2)

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

class Handler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path == '/api/data':
            rows = load_csv()
            verified = load_verified()
            matched = load_matched()
            wrong_rows = [r for r in rows if verified.get(r['ID']) == 'wrong']

            # Get IDs that have been matched (as target) and IDs whose pairs have been used
            matched_targets = set(m['row_id'] for m in matched.get('matches', []))
            used_pair_ids = set(m['from_id'] for m in matched.get('matches', []))

            # Unmatched rows: wrong rows not yet matched as targets
            unmatched_rows = [r for r in wrong_rows if r['ID'] not in matched_targets]
            # Matched rows: wrong rows that have been matched
            matched_rows = [r for r in wrong_rows if r['ID'] in matched_targets]

            # Available pairs: from wrong rows whose pairs haven't been used yet
            pairs = []
            for r in wrong_rows:
                if r['ID'] not in used_pair_ids:
                    # Get original images if this row was matched (use original), else current
                    original = matched.get('original', {}).get(r['ID'])
                    if original:
                        pairs.append({
                            'id': r['ID'],
                            'pic_base': original['pic_base'],
                            'pic_flip': original['pic_flip']
                        })
                    else:
                        pairs.append({
                            'id': r['ID'],
                            'pic_base': r.get('pic_base', ''),
                            'pic_flip': r.get('pic_flip', '')
                        })

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "rows": unmatched_rows,
                "pairs": pairs,
                "matched": matched.get('matches', []),
                "matched_rows": matched_rows
            }).encode())
        elif self.path.startswith('/image/'):
            img_path = urllib.parse.unquote(self.path[7:])
            img_path = img_path.strip().strip("'").strip('"')
            if os.path.exists(img_path):
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Cache-Control', 'public, max-age=86400')
                self.end_headers()
                with open(img_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode()
        data = json.loads(body)

        if self.path == '/api/assign':
            rows = load_csv()
            matched = load_matched()

            target_id = data['target_id']
            pic_base = data['pic_base']
            pic_flip = data['pic_flip']
            from_id = data.get('from_id', '')

            # Save original state before changing
            for row in rows:
                if row['ID'] == target_id:
                    if target_id not in matched.get('original', {}):
                        matched.setdefault('original', {})[target_id] = {
                            'pic_base': row['pic_base'],
                            'pic_flip': row['pic_flip']
                        }
                    row['pic_base'] = pic_base
                    row['pic_flip'] = pic_flip
                    break

            # Record the match
            matched.setdefault('matches', []).append({
                'row_id': target_id,
                'from_id': from_id,
                'pic_base': pic_base,
                'pic_flip': pic_flip
            })

            save_csv(rows)
            save_matched(matched)

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode())

        elif self.path == '/api/unmatch':
            rows = load_csv()
            matched = load_matched()

            row_id = data['row_id']

            # Find and remove the match
            matches = matched.get('matches', [])
            match_to_remove = None
            for m in matches:
                if m['row_id'] == row_id:
                    match_to_remove = m
                    break

            if match_to_remove:
                matches.remove(match_to_remove)

                # Restore original
                original = matched.get('original', {}).get(row_id)
                if original:
                    for row in rows:
                        if row['ID'] == row_id:
                            row['pic_base'] = original['pic_base']
                            row['pic_flip'] = original['pic_flip']
                            break
                    del matched['original'][row_id]

                save_csv(rows)
                save_matched(matched)

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode())

        elif self.path == '/api/update':
            rows = load_csv()
            for row in rows:
                if row['ID'] == data['ID']:
                    for k in ['base_setup', 'base_question', 'flip_question', 'flip_change']:
                        if k in data:
                            row[k] = data[k]
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
    <title>Fix Wrong Rows - Match Pairs</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: system-ui; margin: 0; padding: 20px; background: #1a1a1a; color: #fff; }
        h1 { margin: 0 0 10px 0; }
        h2 { margin: 20px 0 10px 0; font-size: 16px; color: #888; }
        .info { color: #888; margin-bottom: 20px; }

        .container { display: flex; gap: 20px; }

        .pairs-pool {
            width: 280px;
            background: #2a2a2a;
            border-radius: 8px;
            padding: 15px;
            max-height: calc(100vh - 150px);
            overflow-y: auto;
            contain: layout;
        }
        .pairs-pool h3 { margin: 0 0 15px 0; font-size: 14px; color: #888; }

        .pair {
            background: #333;
            border: 2px solid #444;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
            cursor: pointer;
        }
        .pair:hover { border-color: #666; }
        .pair.selected { border-color: #4f4; background: #2a3a2a; }
        .pair-images { display: flex; gap: 8px; }
        .pair-images img { width: 100px; height: 75px; object-fit: cover; border-radius: 4px; }
        .pair-label { font-size: 10px; color: #888; margin-top: 3px; text-align: center; }
        .pair-from { font-size: 10px; color: #666; margin-top: 5px; }

        .main-area { flex: 1; max-height: calc(100vh - 150px); overflow-y: auto; contain: layout; }

        .section { margin-bottom: 30px; }
        .section-title {
            font-size: 14px;
            color: #888;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid #444;
        }
        .section-title span {
            background: #444;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 12px;
            margin-left: 10px;
        }

        .row {
            background: #2a2a2a;
            border: 2px solid #444;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
            display: grid;
            grid-template-columns: 1fr 280px auto;
            gap: 15px;
            align-items: center;
            cursor: pointer;
        }
        .row:hover { border-color: #666; }
        .row.matched-row { border-color: #484; background: #1a2a1a; cursor: default; }

        .row-info h3 { margin: 0 0 5px 0; font-size: 14px; }
        .row-info .setup { color: #888; font-size: 12px; margin-bottom: 5px; }
        .row-info .question { font-size: 13px; }

        .row-images { display: flex; gap: 8px; }
        .row-images img { width: 100px; height: 75px; object-fit: cover; border-radius: 4px; }
        .row-images p { margin: 2px 0 0 0; font-size: 9px; color: #666; text-align: center; }

        .unmatch-btn {
            padding: 8px 12px;
            background: #a44;
            border: none;
            color: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }
        .unmatch-btn:hover { background: #c55; }

        .status {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: #4f4;
            color: #000;
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: bold;
            display: none;
        }
        .status.show { display: block; }

        .nav { margin-bottom: 15px; }
        .nav button { padding: 8px 16px; margin-right: 10px; background: #444; color: #fff; border: none; border-radius: 4px; cursor: pointer; }

        .empty { color: #666; padding: 20px; text-align: center; }

        .modal-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.8);
            z-index: 100;
            align-items: center;
            justify-content: center;
        }
        .modal-overlay.show { display: flex; }
        .modal {
            background: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            width: 600px;
            max-height: 80vh;
            overflow-y: auto;
        }
        .modal h3 { margin: 0 0 15px 0; }
        .modal-field { margin-bottom: 12px; }
        .modal-field label { display: block; font-size: 12px; color: #888; margin-bottom: 4px; }
        .modal-field input, .modal-field textarea {
            width: 100%;
            padding: 8px;
            border: 1px solid #555;
            border-radius: 4px;
            background: #333;
            color: #fff;
            font-size: 14px;
            box-sizing: border-box;
        }
        .modal-field textarea { min-height: 60px; resize: vertical; }
        .modal-buttons { display: flex; gap: 10px; margin-top: 15px; }
        .modal-buttons button { padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
        .btn-save { background: #4a4; color: white; }
        .btn-cancel { background: #666; color: white; }
        .edit-btn { padding: 6px 10px; background: #558; border: none; color: white; border-radius: 4px; cursor: pointer; font-size: 11px; }
        .edit-btn:hover { background: #66a; }
    </style>
</head>
<body>
    <h1>Fix Wrong Rows - Match Image Pairs</h1>
    <p class="info">Click a pair on the left, then click an unmatched row to assign it.</p>
    <div class="nav">
        <button onclick="location.href='http://localhost:8765'">Back to Verify</button>
        <button onclick="load()">Refresh</button>
    </div>

    <div class="container">
        <div class="pairs-pool">
            <h3>Available Pairs</h3>
            <div id="pairs"></div>
        </div>

        <div class="main-area">
            <div class="section">
                <div class="section-title">Unmatched Rows <span id="unmatched-count">0</span></div>
                <div id="unmatched-rows"></div>
            </div>

            <div class="section">
                <div class="section-title">Matched Rows <span id="matched-count">0</span></div>
                <div id="matched-rows"></div>
            </div>
        </div>
    </div>

    <div class="status" id="status"></div>

    <div class="modal-overlay" id="modal-overlay" onclick="closeModal(event)">
        <div class="modal" onclick="event.stopPropagation()">
            <h3>Edit Row: <span id="modal-id"></span></h3>
            <div class="modal-field">
                <label>Base Setup</label>
                <input type="text" id="edit-base_setup">
            </div>
            <div class="modal-field">
                <label>Base Question</label>
                <textarea id="edit-base_question"></textarea>
            </div>
            <div class="modal-field">
                <label>Flip Question</label>
                <textarea id="edit-flip_question"></textarea>
            </div>
            <div class="modal-field">
                <label>Flip Change</label>
                <input type="text" id="edit-flip_change">
            </div>
            <div class="modal-buttons">
                <button class="btn-save" onclick="saveEdit()">Save</button>
                <button class="btn-cancel" onclick="document.getElementById('modal-overlay').classList.remove('show')">Cancel</button>
            </div>
        </div>
    </div>

    <script>
        let rows = [];
        let pairs = [];
        let matchedData = [];
        let matchedRows = [];
        let selectedPair = null;

        function cleanPath(p) {
            return (p || '').trim().replace(/^'+|'+$/g, '').replace(/^"+|"+$/g, '');
        }

        function getFilename(p) {
            return cleanPath(p).split('/').pop();
        }

        async function load() {
            const res = await fetch('/api/data');
            const data = await res.json();
            rows = data.rows;
            pairs = data.pairs;
            matchedData = data.matched || [];
            matchedRows = data.matched_rows || [];
            selectedPair = null;
            render();
        }

        function render() {
            renderPairs();
            renderUnmatchedRows();
            renderMatchedRows();
            document.getElementById('unmatched-count').textContent = rows.length;
            document.getElementById('matched-count').textContent = matchedRows.length;
        }

        function renderPairs() {
            if (pairs.length === 0) {
                document.getElementById('pairs').innerHTML = '<div class="empty">No pairs left!</div>';
                return;
            }
            const html = pairs.map((p, idx) => {
                const base = cleanPath(p.pic_base);
                const flip = cleanPath(p.pic_flip);
                const sel = selectedPair === idx ? 'selected' : '';
                return `
                <div class="pair ${sel}" onclick="selectPair(${idx})">
                    <div class="pair-images">
                        <div>
                            <img loading="lazy" src="/image/${encodeURIComponent(base)}" alt="base">
                            <div class="pair-label">BASE</div>
                        </div>
                        <div>
                            <img loading="lazy" src="/image/${encodeURIComponent(flip)}" alt="flip">
                            <div class="pair-label">FLIP</div>
                        </div>
                    </div>
                    <div class="pair-from">From: ${p.id}</div>
                </div>`;
            }).join('');
            document.getElementById('pairs').innerHTML = html;
        }

        function renderUnmatchedRows() {
            if (rows.length === 0) {
                document.getElementById('unmatched-rows').innerHTML = '<div class="empty">All rows matched!</div>';
                return;
            }
            const html = rows.map(row => {
                const base = cleanPath(row.pic_base);
                const flip = cleanPath(row.pic_flip);
                return `
                <div class="row" onclick="assignToRow('${row.ID}')">
                    <div class="row-info">
                        <h3>${row.ID}</h3>
                        <div class="setup">${row.base_setup || ''}</div>
                        <div class="question"><strong>Q:</strong> ${row.base_question || ''}</div>
                    </div>
                    <div class="row-images">
                        <div>
                            <img loading="lazy" src="/image/${encodeURIComponent(base)}" alt="base">
                        </div>
                        <div>
                            <img loading="lazy" src="/image/${encodeURIComponent(flip)}" alt="flip">
                        </div>
                    </div>
                    <button class="edit-btn" onclick="event.stopPropagation(); openEdit('${row.ID}')">Edit</button>
                </div>`;
            }).join('');
            document.getElementById('unmatched-rows').innerHTML = html;
        }

        function renderMatchedRows() {
            if (matchedRows.length === 0) {
                document.getElementById('matched-rows').innerHTML = '<div class="empty">No matches yet</div>';
                return;
            }
            const html = matchedRows.map(row => {
                const base = cleanPath(row.pic_base);
                const flip = cleanPath(row.pic_flip);
                const match = matchedData.find(m => m.row_id === row.ID);
                return `
                <div class="row matched-row">
                    <div class="row-info">
                        <h3>${row.ID}</h3>
                        <div class="setup">${row.base_setup || ''}</div>
                        <div class="question"><strong>Q:</strong> ${row.base_question || ''}</div>
                    </div>
                    <div class="row-images">
                        <div>
                            <img loading="lazy" src="/image/${encodeURIComponent(base)}" alt="base">
                        </div>
                        <div>
                            <img loading="lazy" src="/image/${encodeURIComponent(flip)}" alt="flip">
                        </div>
                    </div>
                    <div>
                        <button class="edit-btn" onclick="openEdit('${row.ID}')" style="margin-bottom:5px;">Edit</button>
                        <button class="unmatch-btn" onclick="unmatch('${row.ID}')">Undo</button>
                    </div>
                </div>`;
            }).join('');
            document.getElementById('matched-rows').innerHTML = html;
        }

        function selectPair(idx) {
            selectedPair = selectedPair === idx ? null : idx;
            renderPairs();
        }

        async function assignToRow(rowId) {
            if (selectedPair === null) {
                showStatus('Select a pair first!', '#a44');
                return;
            }

            const pair = pairs[selectedPair];
            const row = rows.find(r => r.ID === rowId);
            if (!row) return;

            // Optimistic local update
            const matchInfo = {
                row_id: rowId,
                from_id: pair.id,
                pic_base: pair.pic_base,
                pic_flip: pair.pic_flip
            };

            // Move row to matched
            rows = rows.filter(r => r.ID !== rowId);
            row.pic_base = pair.pic_base;
            row.pic_flip = pair.pic_flip;
            matchedRows.push(row);
            matchedData.push(matchInfo);

            // Remove used pair
            pairs = pairs.filter(p => p.id !== pair.id);
            selectedPair = null;

            render();
            showStatus(`Matched pair to ${rowId}`, '#4f4');

            // Save to server in background
            fetch('/api/assign', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(matchInfo)
            });
        }

        async function unmatch(rowId) {
            const match = matchedData.find(m => m.row_id === rowId);
            const row = matchedRows.find(r => r.ID === rowId);
            if (!row || !match) return;

            // Optimistic local update - move back to unmatched
            matchedRows = matchedRows.filter(r => r.ID !== rowId);
            matchedData = matchedData.filter(m => m.row_id !== rowId);
            rows.push(row);

            // Add pair back
            pairs.push({
                id: match.from_id,
                pic_base: match.pic_base,
                pic_flip: match.pic_flip
            });

            render();
            showStatus(`Unmatched ${rowId}`, '#f84');

            // Save to server in background
            fetch('/api/unmatch', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ row_id: rowId })
            });
        }

        function showStatus(msg, color) {
            const el = document.getElementById('status');
            el.textContent = msg;
            el.style.background = color;
            el.classList.add('show');
            setTimeout(() => el.classList.remove('show'), 2000);
        }

        let editingId = null;

        function openEdit(id) {
            const row = rows.find(r => r.ID === id) || matchedRows.find(r => r.ID === id);
            if (!row) return;
            editingId = id;
            document.getElementById('modal-id').textContent = id;
            document.getElementById('edit-base_setup').value = row.base_setup || '';
            document.getElementById('edit-base_question').value = row.base_question || '';
            document.getElementById('edit-flip_question').value = row.flip_question || '';
            document.getElementById('edit-flip_change').value = row.flip_change || '';
            document.getElementById('modal-overlay').classList.add('show');
        }

        function closeModal(e) {
            if (e.target.id === 'modal-overlay') {
                document.getElementById('modal-overlay').classList.remove('show');
            }
        }

        async function saveEdit() {
            if (!editingId) return;
            const updates = {
                ID: editingId,
                base_setup: document.getElementById('edit-base_setup').value,
                base_question: document.getElementById('edit-base_question').value,
                flip_question: document.getElementById('edit-flip_question').value,
                flip_change: document.getElementById('edit-flip_change').value
            };

            // Update local state
            let row = rows.find(r => r.ID === editingId) || matchedRows.find(r => r.ID === editingId);
            if (row) {
                row.base_setup = updates.base_setup;
                row.base_question = updates.base_question;
                row.flip_question = updates.flip_question;
                row.flip_change = updates.flip_change;
            }

            document.getElementById('modal-overlay').classList.remove('show');
            render();
            showStatus('Saved!', '#4f4');

            // Save to server
            await fetch('/api/update', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(updates)
            });
        }

        load();
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    port = 8766
    print(f"Starting matcher at http://localhost:{port}")
    HTTPServer(('localhost', port), Handler).serve_forever()
