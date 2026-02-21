#!/usr/bin/env python3
"""Image-pair-to-row assignment tool.

On first run, snapshots every row's current (pic_base, pic_flip) into
pairs.json, then clears pic_base/pic_flip in the CSV.  The UI shows
rows on the right and unassigned pairs on the left; click a pair then
click a row to assign it.  Undo is supported.
"""

import csv
import os
import shutil
import subprocess
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import urllib.parse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Sheets", "results.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "Images")
THUMBS_DIR = os.path.join(BASE_DIR, ".thumbs")
PAIRS_PATH = os.path.join(BASE_DIR, "Sheets", "pairs.json")

FIELDNAMES = ['ID', 'Status', 'base_setup', 'base_question', 'flip_question',
              'flip_change', 'pic_base', 'pic_flip', 'indoor_outdoor']

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_csv():
    rows = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

def save_csv(rows):
    with open(CSV_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in FIELDNAMES})

# ---------------------------------------------------------------------------
# Pairs snapshot
# ---------------------------------------------------------------------------

def fname(raw):
    """Extract bare filename from messy path."""
    return os.path.basename(
        (raw or '').strip().strip("'").strip('"')
    )

def load_pairs():
    if os.path.exists(PAIRS_PATH):
        with open(PAIRS_PATH, 'r') as f:
            return json.load(f)
    return None

def save_pairs(data):
    with open(PAIRS_PATH, 'w') as f:
        json.dump(data, f, indent=2)

def snapshot_pairs():
    """First run: extract pairs from CSV, save to pairs.json, clear CSV image cols."""
    existing = load_pairs()
    if existing is not None:
        return existing

    rows = load_csv()
    pairs = []
    for row in rows:
        base = fname(row.get('pic_base', ''))
        flip = fname(row.get('pic_flip', ''))
        if base or flip:
            pairs.append({
                "id": len(pairs),
                "pic_base": base,
                "pic_flip": flip,
                "original_row": row['ID'],
                "assigned_to": None
            })
    data = {"pairs": pairs}
    save_pairs(data)

    # Clear image columns in CSV
    for row in rows:
        row['pic_base'] = ''
        row['pic_flip'] = ''
    save_csv(rows)

    return data

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def resolve_image_path(raw_path):
    p = raw_path.strip().strip("'").strip('"')
    if os.path.isfile(p):
        return p
    f = os.path.basename(p)
    for d in [IMAGES_DIR,
              os.path.join(BASE_DIR, "Images", "Images"),
              os.path.join(BASE_DIR, "Images Sheet Versions")]:
        c = os.path.join(d, f)
        if os.path.isfile(c):
            return c
    return None

def ensure_thumbnail(f):
    os.makedirs(THUMBS_DIR, exist_ok=True)
    thumb = os.path.join(THUMBS_DIR, f)
    if os.path.isfile(thumb):
        return thumb
    src = os.path.join(IMAGES_DIR, f)
    if not os.path.isfile(src):
        return None
    try:
        subprocess.run(['sips', '-Z', '200', src, '--out', thumb],
                       capture_output=True, timeout=10)
        if os.path.isfile(thumb):
            return thumb
    except Exception:
        pass
    return None

def prewarm_thumbnails():
    if not os.path.isdir(IMAGES_DIR):
        return
    for f in sorted(os.listdir(IMAGES_DIR)):
        if f.lower().endswith(('.jpeg', '.jpg', '.png')):
            ensure_thumbnail(f)

# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _serve_file(self, path):
        if path and os.path.isfile(path):
            self.send_response(200)
            self.send_header('Content-type', 'image/jpeg')
            self.send_header('Cache-Control', 'public, max-age=86400')
            self.end_headers()
            with open(path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        p = urllib.parse.urlparse(self.path).path

        if p == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML.encode())

        elif p == '/api/data':
            rows = load_csv()
            pairs_data = load_pairs() or {"pairs": []}
            self._json({"rows": rows, "pairs": pairs_data["pairs"]})

        elif p.startswith('/image/'):
            raw = urllib.parse.unquote(p[7:])
            self._serve_file(resolve_image_path(raw))

        elif p.startswith('/thumb/'):
            f = urllib.parse.unquote(p[7:])
            self._serve_file(ensure_thumbnail(f))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length).decode())

        if self.path == '/api/assign':
            pair_id = body['pair_id']
            row_id = body['row_id']

            pairs_data = load_pairs()
            pair = next((p for p in pairs_data['pairs'] if p['id'] == pair_id), None)
            if not pair:
                self._json({"error": "pair not found"}, 400)
                return

            # Update pair assignment
            pair['assigned_to'] = row_id
            save_pairs(pairs_data)

            # Update CSV
            rows = load_csv()
            for row in rows:
                if row['ID'] == row_id:
                    row['pic_base'] = './Images/' + pair['pic_base']
                    row['pic_flip'] = './Images/' + pair['pic_flip']
                    break
            save_csv(rows)
            self._json({"ok": True})

        elif self.path == '/api/unassign':
            pair_id = body['pair_id']

            pairs_data = load_pairs()
            pair = next((p for p in pairs_data['pairs'] if p['id'] == pair_id), None)
            if not pair or not pair['assigned_to']:
                self._json({"error": "not assigned"}, 400)
                return

            old_row_id = pair['assigned_to']
            pair['assigned_to'] = None
            save_pairs(pairs_data)

            # Clear CSV
            rows = load_csv()
            for row in rows:
                if row['ID'] == old_row_id:
                    row['pic_base'] = ''
                    row['pic_flip'] = ''
                    break
            save_csv(rows)
            self._json({"ok": True})

        elif self.path == '/api/update-row':
            row_id = body.get('id')
            rows = load_csv()
            for row in rows:
                if row['ID'] == row_id:
                    for k in ['base_setup', 'base_question', 'flip_question', 'flip_change', 'indoor_outdoor']:
                        if k in body:
                            row[k] = body[k]
                    break
            save_csv(rows)
            self._json({"ok": True})

        elif self.path == '/api/reset':
            # Reset everything: delete pairs.json, restore backup
            bak = CSV_PATH + '.bak'
            if os.path.isfile(bak):
                shutil.copy2(bak, CSV_PATH)
            if os.path.isfile(PAIRS_PATH):
                os.remove(PAIRS_PATH)
            snapshot_pairs()
            self._json({"ok": True})

        else:
            self.send_response(404)
            self.end_headers()

# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

HTML = r'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Assign Image Pairs</title>
<style>
*, *::before, *::after { box-sizing: border-box; }
body {
    font-family: system-ui, -apple-system, sans-serif;
    margin: 0; padding: 0;
    background: #1a1a1a; color: #e0e0e0;
    height: 100vh; overflow: hidden;
    display: flex; flex-direction: column;
}

/* Header */
.header {
    background: #222; border-bottom: 1px solid #444;
    padding: 10px 16px;
    display: flex; align-items: center; gap: 16px;
    flex-shrink: 0;
}
.header h1 { margin: 0; font-size: 18px; }
.header .stats { font-size: 13px; color: #888; margin-left: auto; }
.header button {
    padding: 6px 14px; border: none; border-radius: 4px;
    font-size: 12px; cursor: pointer;
}
.header .btn-reset { background: #a44; color: #fff; }
.header .btn-reset:hover { background: #c55; }

/* Main two-panel layout */
.container {
    flex: 1; display: flex; overflow: hidden;
}

/* Left: image pairs */
.pairs-panel {
    width: 420px; min-width: 420px;
    background: #1e1e1e; border-right: 1px solid #444;
    display: flex; flex-direction: column;
    overflow: hidden;
}
.pairs-header {
    padding: 10px 14px; border-bottom: 1px solid #333;
    font-size: 13px; color: #888;
    display: flex; align-items: center; gap: 10px;
    flex-shrink: 0;
}
.pairs-header input {
    flex: 1; padding: 5px 8px; border: 1px solid #555; border-radius: 4px;
    background: #333; color: #fff; font-size: 12px;
}
.pairs-list {
    flex: 1; overflow-y: auto; padding: 8px;
}

.pair-card {
    background: #2a2a2a; border: 2px solid #444; border-radius: 8px;
    padding: 10px; margin-bottom: 8px; cursor: pointer;
    transition: border-color 0.15s;
}
.pair-card:hover { border-color: #666; }
.pair-card.selected { border-color: #68f; background: #1e2a3a; }
.pair-card.assigned { border-color: #4a4; background: #1a2a1a; opacity: 0.5; cursor: default; }
.pair-card.assigned:hover { border-color: #4a4; }

.pair-images { display: flex; gap: 6px; }
.pair-images img {
    width: 140px; height: 105px; object-fit: cover; border-radius: 4px;
    flex-shrink: 0; cursor: zoom-in;
}
.pair-meta {
    margin-top: 6px; display: flex; justify-content: space-between; align-items: center;
}
.pair-label { font-size: 10px; color: #666; }
.pair-from { font-size: 10px; color: #555; }
.pair-assigned-to { font-size: 10px; color: #4a4; font-weight: 600; }

.unassign-btn {
    padding: 3px 8px; font-size: 10px;
    background: #a44; color: white; border: none; border-radius: 3px; cursor: pointer;
}
.unassign-btn:hover { background: #c55; }

/* Right: rows */
.rows-panel {
    flex: 1; display: flex; flex-direction: column; overflow: hidden;
}
.rows-header {
    padding: 10px 14px; border-bottom: 1px solid #333;
    font-size: 13px; color: #888;
    display: flex; align-items: center; gap: 10px;
    flex-shrink: 0;
}
.rows-header input {
    flex: 1; padding: 5px 8px; border: 1px solid #555; border-radius: 4px;
    background: #333; color: #fff; font-size: 12px;
}
.rows-header select {
    padding: 5px 8px; border: 1px solid #555; border-radius: 4px;
    background: #333; color: #fff; font-size: 12px;
}
.rows-list {
    flex: 1; overflow-y: auto; padding: 8px;
}

.row-card {
    background: #2a2a2a; border: 2px solid #444; border-radius: 8px;
    padding: 12px; margin-bottom: 6px; cursor: pointer;
    transition: border-color 0.15s;
    display: grid; grid-template-columns: 60px 1fr auto; gap: 12px; align-items: center;
}
.row-card:hover { border-color: #666; }
.row-card.has-pair { border-color: #4a4; background: #1a2a1a; }
.row-card.target { border-color: #68f; background: #1e2a3a; }

.row-id { font-weight: 700; font-size: 14px; color: #fff; }
.row-text { font-size: 12px; color: #aaa; line-height: 1.4; }
.row-text .setup { color: #ccc; font-weight: 500; margin-bottom: 2px; }
.row-text .q { color: #999; }
.row-text .q b { color: #bbb; font-weight: 500; }

.row-pair-preview {
    display: flex; gap: 4px; align-items: center;
}
.row-pair-preview img {
    width: 60px; height: 45px; object-fit: cover; border-radius: 3px;
}
.row-pair-preview .no-pair {
    width: 124px; height: 45px;
    border: 2px dashed #555; border-radius: 3px;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; color: #555;
}

/* Toast */
#toast {
    position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
    background: #4a4; color: #fff;
    padding: 10px 24px; border-radius: 6px;
    font-size: 14px; font-weight: 500;
    opacity: 0; transition: opacity 0.3s;
    z-index: 200; pointer-events: none;
}
#toast.show { opacity: 1; }

/* Instruction banner */
.banner {
    position: fixed; top: 60px; left: 50%; transform: translateX(-50%);
    background: #335; color: #aaf; border: 1px solid #446;
    padding: 8px 20px; border-radius: 6px;
    font-size: 13px; z-index: 50;
    opacity: 0; transition: opacity 0.3s; pointer-events: none;
}
.banner.show { opacity: 1; }

/* Filter tabs */
.filter-tabs { display: flex; gap: 4px; }
.filter-tabs button {
    padding: 4px 10px; border: 1px solid #555; border-radius: 4px;
    background: #333; color: #aaa; font-size: 11px; cursor: pointer;
}
.filter-tabs button.active { background: #446; color: #fff; border-color: #668; }

/* Lightbox */
.lightbox {
    display: none; position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.92);
    z-index: 300;
    align-items: center; justify-content: center;
    flex-direction: column; gap: 16px;
    cursor: zoom-out;
}
.lightbox.show { display: flex; }
.lightbox-images {
    display: flex; gap: 20px; align-items: center; justify-content: center;
    max-width: 95vw;
}
.lightbox-slot { text-align: center; }
.lightbox-slot img {
    max-height: 75vh; max-width: 45vw; object-fit: contain;
    border-radius: 8px; border: 2px solid #555;
}
.lightbox-slot .lb-label {
    margin-top: 8px; font-size: 13px; color: #888;
    text-transform: uppercase; letter-spacing: 1px;
}
.lightbox-slot .lb-fname {
    font-size: 11px; color: #666; margin-top: 2px;
}

/* Edit modal */
.edit-overlay {
    display: none; position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.8);
    z-index: 250;
    align-items: center; justify-content: center;
}
.edit-overlay.show { display: flex; }
.edit-modal {
    background: #2a2a2a; border-radius: 10px;
    padding: 24px; width: 560px; max-height: 85vh; overflow-y: auto;
}
.edit-modal h3 { margin: 0 0 16px 0; font-size: 16px; }
.edit-field { margin-bottom: 12px; }
.edit-field label { display: block; font-size: 11px; color: #888; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
.edit-field input, .edit-field textarea, .edit-field select {
    width: 100%; padding: 8px 10px; border: 1px solid #555; border-radius: 4px;
    background: #333; color: #fff; font-size: 14px; font-family: inherit;
}
.edit-field textarea { min-height: 70px; resize: vertical; }
.edit-field input:focus, .edit-field textarea:focus { border-color: #68f; outline: none; }
.edit-buttons { display: flex; gap: 10px; margin-top: 16px; }
.edit-buttons button { padding: 8px 20px; border: none; border-radius: 4px; font-size: 13px; cursor: pointer; }
.btn-save-edit { background: #4a4; color: white; }
.btn-save-edit:hover { background: #5b5; }
.btn-cancel-edit { background: #555; color: white; }
.btn-cancel-edit:hover { background: #666; }

/* Edit button on row card */
.row-edit-btn {
    padding: 4px 10px; font-size: 11px;
    background: #446; color: #ccc; border: 1px solid #557;
    border-radius: 3px; cursor: pointer; white-space: nowrap;
}
.row-edit-btn:hover { background: #557; color: #fff; }
</style>
</head>
<body>

<div class="header">
    <h1>Assign Image Pairs to Rows</h1>
    <span class="stats" id="stats"></span>
    <button class="btn-reset" onclick="resetAll()">Reset All</button>
</div>

<div class="container">
    <div class="pairs-panel">
        <div class="pairs-header">
            <span>Pairs</span>
            <input type="text" id="pair-search" placeholder="Filter pairs..." oninput="renderPairs()">
            <div class="filter-tabs">
                <button class="active" id="pf-all" onclick="setPairFilter('all')">All</button>
                <button id="pf-free" onclick="setPairFilter('free')">Free</button>
                <button id="pf-done" onclick="setPairFilter('done')">Done</button>
            </div>
        </div>
        <div class="pairs-list" id="pairs-list"></div>
    </div>

    <div class="rows-panel">
        <div class="rows-header">
            <span>Rows</span>
            <input type="text" id="row-search" placeholder="Search rows..." oninput="renderRows()">
            <select id="row-prefix" onchange="renderRows()">
                <option value="">All prefixes</option>
            </select>
            <div class="filter-tabs">
                <button class="active" id="rf-all" onclick="setRowFilter('all')">All</button>
                <button id="rf-empty" onclick="setRowFilter('empty')">Unassigned</button>
                <button id="rf-done" onclick="setRowFilter('done')">Assigned</button>
            </div>
        </div>
        <div class="rows-list" id="rows-list"></div>
    </div>
</div>

<div id="toast"></div>
<div class="banner" id="banner">Now click a row to assign this pair</div>

<div class="lightbox" id="lightbox" onclick="closeLightbox()">
    <div class="lightbox-images" onclick="event.stopPropagation()">
        <div class="lightbox-slot">
            <img id="lb-img-base" src="">
            <div class="lb-label">Base</div>
            <div class="lb-fname" id="lb-fname-base"></div>
        </div>
        <div class="lightbox-slot">
            <img id="lb-img-flip" src="">
            <div class="lb-label">Flip</div>
            <div class="lb-fname" id="lb-fname-flip"></div>
        </div>
    </div>
</div>

<div class="edit-overlay" id="edit-overlay" onclick="if(event.target===this)closeEdit()">
    <div class="edit-modal">
        <h3>Edit Row: <span id="edit-row-id"></span></h3>
        <div class="edit-field">
            <label>Base Setup</label>
            <input type="text" id="edit-base_setup">
        </div>
        <div class="edit-field">
            <label>Base Question</label>
            <textarea id="edit-base_question"></textarea>
        </div>
        <div class="edit-field">
            <label>Flip Question</label>
            <textarea id="edit-flip_question"></textarea>
        </div>
        <div class="edit-field">
            <label>Flip Change</label>
            <input type="text" id="edit-flip_change">
        </div>
        <div class="edit-field">
            <label>Indoor / Outdoor</label>
            <select id="edit-indoor_outdoor">
                <option value="indoor">indoor</option>
                <option value="outdoor">outdoor</option>
                <option value="">(blank)</option>
            </select>
        </div>
        <div class="edit-buttons">
            <button class="btn-save-edit" onclick="saveEdit()">Save</button>
            <button class="btn-cancel-edit" onclick="closeEdit()">Cancel</button>
        </div>
    </div>
</div>

<script>
let rows = [];
let pairs = [];
let selectedPairId = null;
let editingRowId = null;
let pairFilter = 'all';
let rowFilter = 'all';

async function init() {
    const res = await fetch('/api/data');
    const data = await res.json();
    rows = data.rows;
    pairs = data.pairs;
    buildPrefixFilter();
    render();
}

function buildPrefixFilter() {
    const prefixes = new Set();
    rows.forEach(r => {
        const m = (r.ID || '').match(/^([A-Z]+)/);
        if (m) prefixes.add(m[1]);
    });
    const sel = document.getElementById('row-prefix');
    [...prefixes].sort().forEach(p => {
        const opt = document.createElement('option');
        opt.value = p; opt.textContent = p;
        sel.appendChild(opt);
    });
}

function render() {
    updateStats();
    renderPairs();
    renderRows();
}

function updateStats() {
    const assigned = pairs.filter(p => p.assigned_to).length;
    const total = pairs.length;
    document.getElementById('stats').textContent =
        `${assigned} / ${total} pairs assigned — ${total - assigned} remaining`;
}

// --- Pairs ---
function setPairFilter(f) {
    pairFilter = f;
    document.querySelectorAll('.pairs-header .filter-tabs button').forEach(b => b.classList.remove('active'));
    document.getElementById('pf-' + (f === 'free' ? 'free' : f === 'done' ? 'done' : 'all')).classList.add('active');
    renderPairs();
}

function renderPairs() {
    const q = document.getElementById('pair-search').value.toLowerCase();
    let filtered = pairs;
    if (pairFilter === 'free') filtered = pairs.filter(p => !p.assigned_to);
    else if (pairFilter === 'done') filtered = pairs.filter(p => p.assigned_to);
    if (q) filtered = filtered.filter(p =>
        p.pic_base.toLowerCase().includes(q) ||
        p.pic_flip.toLowerCase().includes(q) ||
        (p.original_row || '').toLowerCase().includes(q) ||
        (p.assigned_to || '').toLowerCase().includes(q)
    );

    const html = filtered.map(p => {
        const isSelected = p.id === selectedPairId;
        const isAssigned = !!p.assigned_to;
        let cls = 'pair-card';
        if (isSelected) cls += ' selected';
        if (isAssigned) cls += ' assigned';

        return `<div class="${cls}" onclick="selectPair(${p.id}, ${isAssigned})" data-pair-id="${p.id}">
            <div class="pair-images">
                <div style="text-align:center">
                    <img loading="lazy" src="/thumb/${encodeURIComponent(p.pic_base)}"
                         onerror="this.src='/image/'+encodeURIComponent(this.dataset.f)" data-f="${esc(p.pic_base)}"
                         onclick="event.stopPropagation(); openLightbox('${esc(p.pic_base)}','${esc(p.pic_flip)}')">
                    <div class="pair-label">BASE</div>
                </div>
                <div style="text-align:center">
                    <img loading="lazy" src="/thumb/${encodeURIComponent(p.pic_flip)}"
                         onerror="this.src='/image/'+encodeURIComponent(this.dataset.f)" data-f="${esc(p.pic_flip)}"
                         onclick="event.stopPropagation(); openLightbox('${esc(p.pic_base)}','${esc(p.pic_flip)}')">
                    <div class="pair-label">FLIP</div>
                </div>
            </div>
            <div class="pair-meta">
                <span class="pair-from">Was: ${esc(p.original_row)}</span>
                ${isAssigned
                    ? `<span class="pair-assigned-to">→ ${esc(p.assigned_to)}</span>
                       <button class="unassign-btn" onclick="event.stopPropagation(); unassignPair(${p.id})">Undo</button>`
                    : `<span class="pair-label">#${p.id}</span>`
                }
            </div>
        </div>`;
    }).join('');
    document.getElementById('pairs-list').innerHTML = html || '<div style="padding:20px;color:#666;text-align:center">No pairs</div>';
}

function selectPair(id, isAssigned) {
    if (isAssigned) return;
    if (selectedPairId === id) {
        selectedPairId = null;
        document.getElementById('banner').classList.remove('show');
    } else {
        selectedPairId = id;
        document.getElementById('banner').classList.add('show');
    }
    renderPairs();
}

// --- Rows ---
function setRowFilter(f) {
    rowFilter = f;
    document.querySelectorAll('.rows-header .filter-tabs button').forEach(b => b.classList.remove('active'));
    document.getElementById('rf-' + (f === 'empty' ? 'empty' : f === 'done' ? 'done' : 'all')).classList.add('active');
    renderRows();
}

function getAssignedPair(rowId) {
    return pairs.find(p => p.assigned_to === rowId);
}

function renderRows() {
    const q = document.getElementById('row-search').value.toLowerCase();
    const prefix = document.getElementById('row-prefix').value;

    let filtered = rows;
    if (prefix) filtered = filtered.filter(r => r.ID.startsWith(prefix + '-'));
    if (rowFilter === 'empty') filtered = filtered.filter(r => !getAssignedPair(r.ID));
    else if (rowFilter === 'done') filtered = filtered.filter(r => !!getAssignedPair(r.ID));
    if (q) filtered = filtered.filter(r =>
        Object.values(r).join(' ').toLowerCase().includes(q)
    );

    const html = filtered.map(row => {
        const ap = getAssignedPair(row.ID);
        let cls = 'row-card';
        if (ap) cls += ' has-pair';
        if (selectedPairId !== null && !ap) cls += ' target';

        let preview;
        if (ap) {
            preview = `<div class="row-pair-preview">
                <img src="/thumb/${encodeURIComponent(ap.pic_base)}" style="cursor:zoom-in"
                     onclick="event.stopPropagation(); openLightbox('${esc(ap.pic_base)}','${esc(ap.pic_flip)}')">
                <img src="/thumb/${encodeURIComponent(ap.pic_flip)}" style="cursor:zoom-in"
                     onclick="event.stopPropagation(); openLightbox('${esc(ap.pic_base)}','${esc(ap.pic_flip)}')">
                <button class="unassign-btn" onclick="event.stopPropagation(); unassignPair(${ap.id})" style="margin-left:4px">Undo</button>
            </div>`;
        } else {
            preview = `<div class="row-pair-preview"><div class="no-pair">no pair</div></div>`;
        }

        return `<div class="${cls}" onclick="assignToRow('${esc(row.ID)}')">
            <div class="row-id">${esc(row.ID)}</div>
            <div class="row-text">
                <div class="setup">${esc(row.base_setup)}</div>
                <div class="q"><b>Q:</b> ${esc(row.base_question)}</div>
            </div>
            <div style="display:flex;flex-direction:column;gap:4px;align-items:end">
                ${preview}
                <button class="row-edit-btn" onclick="event.stopPropagation(); openEdit('${esc(row.ID)}')">Edit</button>
            </div>
        </div>`;
    }).join('');
    document.getElementById('rows-list').innerHTML = html || '<div style="padding:20px;color:#666;text-align:center">No rows</div>';
}

// --- Assign / Unassign ---
async function assignToRow(rowId) {
    if (selectedPairId === null) return;
    // Don't assign if row already has a pair
    if (getAssignedPair(rowId)) {
        showToast('Row already has a pair — unassign first', true);
        return;
    }

    const pairId = selectedPairId;

    // Optimistic local update
    const pair = pairs.find(p => p.id === pairId);
    if (pair) pair.assigned_to = rowId;
    selectedPairId = null;
    document.getElementById('banner').classList.remove('show');
    render();
    showToast('Assigned pair #' + pairId + ' → ' + rowId);

    await fetch('/api/assign', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({pair_id: pairId, row_id: rowId})
    });
}

async function unassignPair(pairId) {
    const pair = pairs.find(p => p.id === pairId);
    if (!pair) return;

    const oldRow = pair.assigned_to;
    pair.assigned_to = null;
    render();
    showToast('Unassigned pair #' + pairId + ' from ' + oldRow);

    await fetch('/api/unassign', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({pair_id: pairId})
    });
}

async function resetAll() {
    if (!confirm('Reset all assignments and re-extract pairs from backup?')) return;
    await fetch('/api/reset', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
    location.reload();
}

function esc(s) {
    return (s || '').replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function showToast(msg, isError) {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.style.background = isError ? '#a44' : '#4a4';
    t.classList.add('show');
    clearTimeout(showToast._timer);
    showToast._timer = setTimeout(() => t.classList.remove('show'), 1500);
}

// --- Edit modal ---
function openEdit(id) {
    const row = rows.find(r => r.ID === id);
    if (!row) return;
    editingRowId = id;
    document.getElementById('edit-row-id').textContent = id;
    document.getElementById('edit-base_setup').value = row.base_setup || '';
    document.getElementById('edit-base_question').value = row.base_question || '';
    document.getElementById('edit-flip_question').value = row.flip_question || '';
    document.getElementById('edit-flip_change').value = row.flip_change || '';
    document.getElementById('edit-indoor_outdoor').value = row.indoor_outdoor || '';
    document.getElementById('edit-overlay').classList.add('show');
}

function closeEdit() {
    document.getElementById('edit-overlay').classList.remove('show');
    editingRowId = null;
}

async function saveEdit() {
    if (!editingRowId) return;
    const updates = {
        id: editingRowId,
        base_setup: document.getElementById('edit-base_setup').value,
        base_question: document.getElementById('edit-base_question').value,
        flip_question: document.getElementById('edit-flip_question').value,
        flip_change: document.getElementById('edit-flip_change').value,
        indoor_outdoor: document.getElementById('edit-indoor_outdoor').value
    };

    // Update local state
    const row = rows.find(r => r.ID === editingRowId);
    if (row) {
        row.base_setup = updates.base_setup;
        row.base_question = updates.base_question;
        row.flip_question = updates.flip_question;
        row.flip_change = updates.flip_change;
        row.indoor_outdoor = updates.indoor_outdoor;
    }

    closeEdit();
    renderRows();
    showToast('Saved ' + editingRowId);

    await fetch('/api/update-row', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(updates)
    });
}

// --- Lightbox ---
function openLightbox(baseF, flipF) {
    document.getElementById('lb-img-base').src = '/image/' + encodeURIComponent(baseF);
    document.getElementById('lb-img-flip').src = '/image/' + encodeURIComponent(flipF);
    document.getElementById('lb-fname-base').textContent = baseF;
    document.getElementById('lb-fname-flip').textContent = flipF;
    document.getElementById('lightbox').classList.add('show');
}
function closeLightbox() {
    document.getElementById('lightbox').classList.remove('show');
}

// Keyboard: Escape deselects pair / closes lightbox
document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
        if (document.getElementById('lightbox').classList.contains('show')) {
            closeLightbox();
        } else {
            selectedPairId = null;
            document.getElementById('banner').classList.remove('show');
            renderPairs();
        }
    }
});

init();
</script>
</body>
</html>
'''

if __name__ == '__main__':
    # Backup CSV on startup (only if no backup exists yet)
    bak = CSV_PATH + '.bak'
    if not os.path.isfile(bak) and os.path.isfile(CSV_PATH):
        shutil.copy2(CSV_PATH, bak)
        print(f"Backup: {bak}")
    else:
        print(f"Backup already exists: {bak}")

    # Snapshot pairs (first run extracts from CSV; subsequent runs use pairs.json)
    data = snapshot_pairs()
    assigned = sum(1 for p in data['pairs'] if p['assigned_to'])
    total = len(data['pairs'])
    print(f"Pairs: {assigned}/{total} assigned")

    # Pre-generate thumbnails in background
    os.makedirs(THUMBS_DIR, exist_ok=True)
    threading.Thread(target=prewarm_thumbnails, daemon=True).start()

    port = 8767
    print(f"Starting at http://localhost:{port}")
    HTTPServer(('localhost', port), Handler).serve_forever()
