"""
Pre-push security check — run before pushing to GitHub.
Scans for leaked API keys, secrets, and sensitive data.

Usage: python security_check.py
"""
import os
import re
import sys

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

SECRET_PATTERNS = [
    (r'AIzaSy[A-Za-z0-9_-]{33}', 'Google API Key'),
    (r'sk-ant-[A-Za-z0-9_-]{40,}', 'Anthropic API Key'),
    (r'sk-[A-Za-z0-9]{48,}', 'OpenAI API Key'),
    (r'gsk_[A-Za-z0-9]{40,}', 'Groq API Key'),
    (r'xai-[A-Za-z0-9]{40,}', 'xAI API Key'),
    (r'hf_[A-Za-z0-9]{30,}', 'HuggingFace Token'),
    (r'sk-or-[A-Za-z0-9]{40,}', 'OpenRouter Key'),
    (r'ghp_[A-Za-z0-9]{36}', 'GitHub Personal Token'),
]

SKIP = {'.git', '__pycache__', 'node_modules', '.env', 'security_check.py'}
SKIP_EXT = {'.pyc', '.pyo', '.exe', '.dll', '.so', '.bin', '.jpg', '.png'}


def scan_file(filepath):
    issues = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        for pattern, name in SECRET_PATTERNS:
            for match in re.findall(pattern, content):
                issues.append((filepath, name, match[:20] + '...'))
    except Exception:
        pass
    return issues


def main():
    print("Neural Brain API - Security Scan")
    print("=" * 45)

    all_issues = []
    for root, dirs, files in os.walk(REPO_DIR):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for fname in files:
            if fname in SKIP or os.path.splitext(fname)[1] in SKIP_EXT:
                continue
            all_issues.extend(scan_file(os.path.join(root, fname)))

    env_file = os.path.join(REPO_DIR, '.env')
    gitignore = os.path.join(REPO_DIR, '.gitignore')
    if os.path.exists(env_file):
        if not os.path.exists(gitignore):
            all_issues.append(('.gitignore', 'CRITICAL', 'No .gitignore found!'))
        else:
            with open(gitignore) as f:
                if '.env' not in f.read():
                    all_issues.append(('.gitignore', 'CRITICAL', '.env NOT in .gitignore!'))

    if all_issues:
        print("\nSECRETS FOUND - DO NOT PUSH!\n")
        for filepath, issue_type, preview in all_issues:
            rel = os.path.relpath(filepath, REPO_DIR)
            print(f"  [!] {rel}: {issue_type} ({preview})")
        print(f"\n{len(all_issues)} issue(s) found. Fix before pushing.")
        sys.exit(1)
    else:
        print("\nNo secrets found. Safe to push.")
        sys.exit(0)


if __name__ == "__main__":
    main()
