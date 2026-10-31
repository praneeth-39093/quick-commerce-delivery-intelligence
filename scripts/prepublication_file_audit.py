"""
Audit script for credentials, file sizes, and GitHub readiness.
"""
import os
import re
from pathlib import Path

root = Path(__file__).resolve().parent.parent

secret_patterns = [
    re.compile(r'(?i)(api[_-]?key|secret[_-]?key|auth[_-]?token|private[_-]?key)\s*[:=]\s*[\'\"][^\'\"]+[\'\"]'),
    re.compile(r'(?i)bearer\s+[a-zA-Z0-9_\-\.]{20,}'),
    re.compile(r'(?i)password\s*[:=]\s*[\'\"][^\'\"]+[\'\"]')
]

findings = []
large_files = []
ignored_dirs = {'venv', '.venv', '.git', '.pytest_cache', '__pycache__'}

for p in root.rglob('*'):
    if not p.is_file():
        continue
    if any(part in ignored_dirs for part in p.parts):
        continue
    
    # File size check
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > 25.0:  # Warning above 25MB
        large_files.append((str(p.relative_to(root)), round(size_mb, 2)))

    # Skip binaries for text search
    if p.suffix in ['.joblib', '.png', '.xlsx', '.csv', '.pyc']:
        continue

    try:
        content = p.read_text(encoding='utf-8', errors='ignore')
        for idx, line in enumerate(content.splitlines(), start=1):
            for pat in secret_patterns:
                if pat.search(line):
                    findings.append(f"{p.relative_to(root)}:{idx} -> {line.strip()[:60]}")
    except Exception as e:
        pass

print("=== 1. SECRET & CREDENTIAL AUDIT ===")
if findings:
    print(f"⚠️ {len(findings)} potential secrets found:")
    for f in findings:
        print("  ", f)
else:
    print("✅ Zero secrets, API keys, or private tokens detected in project files.")

print("\n=== 2. FILE SIZE AUDIT (> 25MB) ===")
if large_files:
    print(f"⚠️ {len(large_files)} files exceed 25MB threshold:")
    for path, sz in large_files:
        print(f"  {path}: {sz} MB")
else:
    print("✅ Zero files exceed GitHub's 50MB warning or 100MB hard limit.")

print("\n=== 3. ALL PROJECT MODEL & DATA SIZES ===")
for folder in ['models', 'data']:
    dir_path = root / folder
    if dir_path.exists():
        print(f"\nContents of {folder}/:")
        for item in sorted(dir_path.iterdir()):
            if item.is_file():
                sz_kb = item.stat().st_size / 1024
                print(f"  • {item.name}: {sz_kb:.1f} KB ({sz_kb/1024:.2f} MB)")

