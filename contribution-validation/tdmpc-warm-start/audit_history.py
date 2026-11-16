"""Validation-only public Git history audit; never included in the upstream patch."""

import ast
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def audit() -> dict:
    with tempfile.TemporaryDirectory(prefix="lerobot-tdmpc-history-") as directory:
        repo = Path(directory) / "history.git"
        git_executable = shutil.which("git")
        assert git_executable is not None
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"}

        def git(*args: str, timeout: int = 600) -> bytes:
            result = subprocess.run(
                [git_executable, "-C", str(repo), *args],
                check=True,
                capture_output=True,
                env=env,
                timeout=timeout,
            )
            return result.stdout

        subprocess.run([git_executable, "init", "--bare", str(repo)], check=True, capture_output=True, env=env)
        remotes = {
            "upstream": "https://github.com/huggingface/lerobot.git",
            "fork": "https://github.com/Afloat16/lerobot.git",
        }
        requested = {
            "upstream": [
                ("refs/heads/", "refs/audit-upstream/"),
                ("refs/pull/", "refs/audit-pr/"),
                ("refs/tags/", "refs/audit-upstream-tags/"),
            ],
            "fork": [
                ("refs/heads/", "refs/audit-fork/"),
                ("refs/tags/", "refs/audit-fork-tags/"),
            ],
        }
        references = {}
        for name, url in remotes.items():
            refspecs = []
            for source, destination in requested[name]:
                # Fetch real PR heads, not synthetic merge commits.
                suffix = "*/head" if source == "refs/pull/" else "*"
                refspecs.append(f"+{source}{suffix}:{destination}{suffix}")
            git("fetch", "--no-tags", "--no-recurse-submodules", url, *refspecs, timeout=1200)
            advertised = git(
                "ls-remote",
                "--refs",
                url,
                *[source + ("*/head" if source == "refs/pull/" else "*") for source, _ in requested[name]],
            ).decode()
            for line in advertised.splitlines():
                sha, ref = line.split("\t")
                source, destination = next(pair for pair in requested[name] if ref.startswith(pair[0]))
                target = destination + ref[len(source) :]
                actual = git("rev-parse", target).decode().strip()
                if actual != sha:
                    raise AssertionError(f"History snapshot changed during audit: {ref}: {actual} != {sha}")
                references[f"{name}:{ref}"] = sha

        assert git("rev-parse", "--is-shallow-repository").strip() == b"false"
        fork_heads = sum(key.startswith("fork:refs/heads/") for key in references)
        pr_heads = sum(key.startswith("upstream:refs/pull/") for key in references)
        assert fork_heads >= 261, fork_heads
        assert pr_heads > 1000, pr_heads

        all_names = set(
            git("log", "--all", "--full-history", "--no-renames", "--format=", "--name-only")
            .decode()
            .splitlines()
        )
        paths = {path for path in all_names if "tdmpc" in path.lower() and path.endswith(".py")}
        # Include earlier filenames, even when neither their basename nor their directory says TDMPC.
        rename_log = git(
            "log", "--all", "--full-history", "--find-renames=20%", "--format=", "--name-status"
        ).decode()
        renames = []
        for line in rename_log.splitlines():
            fields = line.split("\t")
            if len(fields) == 3 and fields[0].startswith("R"):
                renames.append((fields[1], fields[2]))
        changed = True
        while changed:
            changed = False
            for old, new in renames:
                if (old in paths or new in paths) and not {old, new}.issubset(paths):
                    paths.update((old, new))
                    changed = True

        raw = git(
            "log",
            "--all",
            "--full-history",
            "--no-renames",
            "--format=",
            "--raw",
            "--no-abbrev",
            "--",
            *sorted(paths),
        ).decode()
        blobs = {}
        for line in raw.splitlines():
            if not line.startswith(":"):
                continue
            header, path = line.split("\t", 1)
            fields = header.split()
            for sha in fields[2:4]:
                if set(sha) != {"0"}:
                    blobs.setdefault(sha, set()).add(path)

        output = git("cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype)")
        object_types = dict(line.split() for line in output.decode().splitlines())
        blob_ids = sorted(sha for sha in blobs if object_types.get(sha) == "blob")
        process = subprocess.run(
            [git_executable, "-C", str(repo), "cat-file", "--batch"],
            input="".join(sha + "\n" for sha in blob_ids).encode(),
            capture_output=True,
            check=True,
            env=env,
            timeout=600,
        )
        content = process.stdout
        cursor = 0
        variants = {}
        source_index = []
        parse_failures = []
        for expected_sha in blob_ids:
            newline = content.index(b"\n", cursor)
            sha, kind, size = content[cursor:newline].decode().split()
            assert sha == expected_sha and kind == "blob"
            start = newline + 1
            end = start + int(size)
            text = content[start:end].decode("utf-8", errors="replace")
            cursor = end + 1
            source_index.append({"blob": sha, "paths": sorted(blobs[sha]), "bytes": int(size)})
            try:
                syntax = ast.parse(text)
            except SyntaxError as error:
                parse_failures.append({"blob": sha, "error": str(error)})
                continue
            lines = text.splitlines()
            for node in ast.walk(syntax):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                source = "\n".join(lines[node.lineno - 1 : node.end_lineno])
                # Inspect every planner variant and every function holding warm-start state,
                # including intermediate and later-reverted commits.
                if (
                    "plan" in node.name.lower()
                    or "_prev_mean" in source
                    or "prev_mean" in source
                    or node.name in ("select_action", "predict_action_chunk")
                ):
                    digest = hashlib.sha256(source.encode()).hexdigest()
                    entry = variants.setdefault(
                        digest,
                        {
                            "function": node.name,
                            "source": source,
                            "blobs": [],
                            "paths": [],
                        },
                    )
                    entry["blobs"].append(sha)
                    entry["paths"] = sorted(set(entry["paths"]).union(blobs[sha]))
        result = {
            "complete": not parse_failures,
            "repository": "huggingface/lerobot",
            "fork": "Afloat16/lerobot",
            "ref_counts": {
                "fork_heads": fork_heads,
                "upstream_pr_heads": pr_heads,
                "all_references": len(references),
            },
            "reachable_commits": int(git("rev-list", "--all", "--count")),
            "references": references,
            "paths": sorted(paths),
            "source_blobs": source_index,
            "function_variants": variants,
            "parse_failures": parse_failures,
        }
        return result
