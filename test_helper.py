#!/usr/bin/env python3
"""
Test helper for comparing merged LoRA models vs base models.

Runs speed benchmarks (tok/s at temp=0) and quality pass-rate tests
(code generation with execution verification) against an OpenAI-compatible
server. Outputs JSON results for later comparison.

Usage (typically called by test_merged_model.sh):
    python3 test_helper.py --server URL --output FILE --model-id ID
    python3 test_helper.py --compare A.json B.json
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPEED_PROMPTS = [
    {
        "name": "fibonacci_memo",
        "messages": [
            {"role": "user", "content": (
                "Write a Python function `fib(n)` that returns the nth Fibonacci "
                "number using memoization with a dictionary. Include type hints. "
                "Only output the code, no explanation."
            )}
        ],
        "max_tokens": 1500,
    },
    {
        "name": "binary_search",
        "messages": [
            {"role": "user", "content": (
                "Write a Python function `binary_search(arr, target)` that returns "
                "the index of target in a sorted list, or -1 if not found. "
                "Only output the code, no explanation."
            )}
        ],
        "max_tokens": 1500,
    },
    {
        "name": "trie_class",
        "messages": [
            {"role": "user", "content": (
                "Write a Python class `Trie` with methods `insert(word)`, "
                "`search(word)` -> bool, and `starts_with(prefix)` -> bool. "
                "Only output the code, no explanation."
            )}
        ],
        "max_tokens": 2000,
    },
]

QUALITY_PROBLEMS = [
    {
        "name": "merge_sorted_lists",
        "prompt": (
            "Write a Python function `merge_sorted_lists(a, b)` that merges two "
            "sorted lists into one sorted list without using sort(). "
            "Only output the Python code, no explanation."
        ),
        "test_code": textwrap.dedent("""\
            assert merge_sorted_lists([1,3,5], [2,4,6]) == [1,2,3,4,5,6]
            assert merge_sorted_lists([], [1,2]) == [1,2]
            assert merge_sorted_lists([1,2], []) == [1,2]
            assert merge_sorted_lists([], []) == []
            assert merge_sorted_lists([1], [1]) == [1,1]
        """),
    },
    {
        "name": "balanced_parens",
        "prompt": (
            "Write a Python function `balanced_parens(n)` that generates all "
            "combinations of n pairs of balanced parentheses. Return a list of "
            "strings. Only output the Python code, no explanation."
        ),
        "test_code": textwrap.dedent("""\
            assert sorted(balanced_parens(2)) == sorted(["(())", "()()"])
            assert len(balanced_parens(3)) == 5
            assert balanced_parens(0) == [""]
            assert balanced_parens(1) == ["()"]
        """),
    },
    {
        "name": "longest_increasing_subsequence",
        "prompt": (
            "Write a Python function `longest_increasing_subsequence(nums)` that "
            "returns the length of the longest strictly increasing subsequence. "
            "Only output the Python code, no explanation."
        ),
        "test_code": textwrap.dedent("""\
            assert longest_increasing_subsequence([10,9,2,5,3,7,101,18]) == 4
            assert longest_increasing_subsequence([0,1,0,3,2,3]) == 4
            assert longest_increasing_subsequence([7,7,7,7]) == 1
            assert longest_increasing_subsequence([1,2,3,4,5]) == 5
            assert longest_increasing_subsequence([]) == 0
        """),
    },
    {
        "name": "matrix_multiply",
        "prompt": (
            "Write a Python function `matrix_multiply(a, b)` that multiplies two "
            "2D lists (matrices) and returns the result as a 2D list. "
            "Only output the Python code, no explanation."
        ),
        "test_code": textwrap.dedent("""\
            assert matrix_multiply([[1,2],[3,4]], [[5,6],[7,8]]) == [[19,22],[43,50]]
            assert matrix_multiply([[1,0],[0,1]], [[5,6],[7,8]]) == [[5,6],[7,8]]
            assert matrix_multiply([[2]], [[3]]) == [[6]]
        """),
    },
    {
        "name": "evaluate_rpn",
        "prompt": (
            "Write a Python function `evaluate_rpn(tokens)` that evaluates a list "
            "of tokens in Reverse Polish Notation. Tokens are strings of integers "
            "or operators (+, -, *, /). Division should truncate toward zero. "
            "Only output the Python code, no explanation."
        ),
        "test_code": textwrap.dedent("""\
            assert evaluate_rpn(["2","1","+","3","*"]) == 9
            assert evaluate_rpn(["4","13","5","/","+"]) == 6
            assert evaluate_rpn(["10","6","9","3","+","-11","*","/","*","17","+","5","+"]) == 22
            assert evaluate_rpn(["3"]) == 3
        """),
    },
    {
        "name": "lru_cache",
        "prompt": (
            "Write a Python class `LRUCache` that implements a least-recently-used "
            "cache. The constructor takes an integer `capacity`. It must support "
            "`get(key) -> int` (returns -1 if key not found) and "
            "`put(key, value) -> None`. Both operations must run in O(1) time. "
            "When the cache exceeds capacity on a put, evict the least recently "
            "used key. Only output the Python code, no explanation."
        ),
        "test_code": textwrap.dedent("""\
            c = LRUCache(2)
            c.put(1, 1)
            c.put(2, 2)
            assert c.get(1) == 1
            c.put(3, 3)
            assert c.get(2) == -1
            c.put(4, 4)
            assert c.get(1) == -1
            assert c.get(3) == 3
            assert c.get(4) == 4
            c2 = LRUCache(1)
            c2.put(1, 10)
            c2.put(1, 20)
            assert c2.get(1) == 20
        """),
    },
    {
        "name": "serialize_binary_tree",
        "prompt": (
            "Write two Python functions: `serialize(root)` that converts a binary "
            "tree to a string, and `deserialize(data)` that reconstructs the tree "
            "from that string. Define a `TreeNode` class with attributes `val`, "
            "`left`, `right` and a constructor `TreeNode(val=0, left=None, right=None)`. "
            "The serialization format is your choice, but `deserialize(serialize(tree))` "
            "must produce an identical tree. Handle None/empty trees. "
            "Only output the Python code, no explanation."
        ),
        "test_code": textwrap.dedent("""\
            r = TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4), TreeNode(5)))
            s = serialize(r)
            r2 = deserialize(s)
            assert r2.val == 1
            assert r2.left.val == 2
            assert r2.left.left is None
            assert r2.right.val == 3
            assert r2.right.left.val == 4
            assert r2.right.right.val == 5
            assert deserialize(serialize(None)) is None
            r3 = deserialize(serialize(TreeNode(42)))
            assert r3.val == 42 and r3.left is None and r3.right is None
        """),
    },
    {
        "name": "valid_ip_addresses",
        "prompt": (
            "Write a Python function `restore_ip_addresses(s)` that takes a string "
            "of digits and returns all possible valid IPv4 addresses that can be "
            "formed by inserting exactly 3 dots. Each octet must be between 0 and "
            "255 and must not have leading zeros (except '0' itself). Return a list "
            "of strings in dotted notation. Only output the Python code, no explanation."
        ),
        "test_code": textwrap.dedent("""\
            r1 = sorted(restore_ip_addresses("25525511135"))
            assert r1 == sorted(["255.255.11.135", "255.255.111.35"])
            r2 = sorted(restore_ip_addresses("0000"))
            assert r2 == ["0.0.0.0"]
            r3 = sorted(restore_ip_addresses("1111"))
            assert r3 == ["1.1.1.1"]
            r4 = restore_ip_addresses("256256256256")
            assert r4 == []
            r5 = sorted(restore_ip_addresses("101023"))
            assert "1.0.10.23" in r5
            assert "10.1.0.23" in r5
            for ip in restore_ip_addresses("010010"):
                for octet in ip.split("."):
                    assert octet == "0" or not octet.startswith("0")
        """),
    },
    {
        "name": "regex_matcher",
        "prompt": (
            "Write a Python function `is_match(s, p)` that implements regular "
            "expression matching supporting '.' (matches any single character) "
            "and '*' (matches zero or more of the preceding element). The match "
            "must cover the entire input string s (not partial). "
            "Only output the Python code, no explanation."
        ),
        "test_code": textwrap.dedent("""\
            assert is_match("aa", "a") == False
            assert is_match("aa", "a*") == True
            assert is_match("ab", ".*") == True
            assert is_match("aab", "c*a*b") == True
            assert is_match("mississippi", "mis*is*ip*.") == True
            assert is_match("mississippi", "mis*is*p*.") == False
            assert is_match("", "a*") == True
            assert is_match("", "") == True
            assert is_match("a", "") == False
            assert is_match("aaa", "a*a") == True
            assert is_match("aaa", "ab*a*c*a") == True
        """),
    },
    {
        "name": "topological_sort",
        "prompt": (
            "Write a Python function `topological_sort(num_nodes, edges)` where "
            "`num_nodes` is an integer and `edges` is a list of [from, to] pairs "
            "representing a directed graph. Return a list of node indices in a valid "
            "topological order. If the graph has a cycle, return an empty list. "
            "Node indices are 0-based. Only output the Python code, no explanation."
        ),
        "test_code": textwrap.dedent("""\
            r = topological_sort(3, [[0,1],[1,2],[0,2]])
            assert len(r) == 3
            assert r.index(0) < r.index(1) < r.index(2)
            assert topological_sort(3, [[0,1],[1,2],[2,0]]) == []
            r2 = topological_sort(4, [])
            assert sorted(r2) == [0,1,2,3]
            r3 = topological_sort(4, [[0,1],[0,2],[1,3],[2,3]])
            assert r3.index(0) < r3.index(1)
            assert r3.index(0) < r3.index(2)
            assert r3.index(1) < r3.index(3)
            assert r3.index(2) < r3.index(3)
            assert topological_sort(1, []) == [0]
        """),
    },
    {
        "name": "min_window_substring",
        "prompt": (
            "Write a Python function `min_window(s, t)` that finds the minimum "
            "window substring of `s` that contains all characters of `t` "
            "(including duplicates). If no such window exists, return an empty "
            "string. If there are multiple minimum-length windows, return the "
            "one that appears first. Only output the Python code, no explanation."
        ),
        "test_code": textwrap.dedent("""\
            assert min_window("ADOBECODEBANC", "ABC") == "BANC"
            assert min_window("a", "a") == "a"
            assert min_window("a", "aa") == ""
            assert min_window("aa", "aa") == "aa"
            assert min_window("bba", "ab") == "ba"
            assert min_window("abc", "d") == ""
        """),
    },
    {
        "name": "merge_sort_inversions",
        "prompt": (
            "Write a Python function `count_inversions(arr)` that returns a tuple "
            "`(sorted_array, count)` where `sorted_array` is the input list sorted "
            "in ascending order and `count` is the number of inversions (pairs "
            "(i,j) where i < j but arr[i] > arr[j]). Use the merge sort algorithm "
            "to achieve O(n log n) time. Only output the Python code, no explanation."
        ),
        "test_code": textwrap.dedent("""\
            s, c = count_inversions([2, 4, 1, 3, 5])
            assert s == [1, 2, 3, 4, 5]
            assert c == 3
            s2, c2 = count_inversions([5, 4, 3, 2, 1])
            assert s2 == [1, 2, 3, 4, 5]
            assert c2 == 10
            s3, c3 = count_inversions([1, 2, 3])
            assert s3 == [1, 2, 3]
            assert c3 == 0
            s4, c4 = count_inversions([1])
            assert s4 == [1]
            assert c4 == 0
            s5, c5 = count_inversions([])
            assert s5 == []
            assert c5 == 0
        """),
    },
    {
        "name": "expression_evaluator",
        "prompt": (
            "Write a Python function `evaluate(expression)` that takes a string "
            "containing an arithmetic expression with integers, +, -, *, /, "
            "parentheses, and spaces. It should respect standard operator precedence "
            "(* and / before + and -) and parentheses. Division should use integer "
            "division truncating toward zero. Return the integer result. "
            "Only output the Python code, no explanation."
        ),
        "test_code": textwrap.dedent("""\
            assert evaluate("3+2*2") == 7
            assert evaluate(" 3/2 ") == 1
            assert evaluate(" 3+5 / 2 ") == 5
            assert evaluate("(1+(4+5+2)-3)+(6+8)") == 23
            assert evaluate("2*(5+5*2)/3+(6/2+8)") == 21
            assert evaluate("1-1+1") == 1
            assert evaluate("0") == 0
        """),
    },
]

QUALITY_SAMPLES = 10
QUALITY_TEMP = 0.7
QUALITY_MAX_TOKENS = 2000
REQUEST_TIMEOUT = 120
_MODEL_ID = "default"

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def chat_completion(server_url: str, messages: list, temperature: float,
                    max_tokens: int) -> dict:
    """Send a chat completion request to an OpenAI-compatible server."""
    payload = json.dumps({
        "model": _MODEL_ID,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        f"{server_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            body = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return {"error": str(e), "elapsed": time.monotonic() - t0}

    body["_elapsed"] = time.monotonic() - t0
    return body


def extract_content(response: dict) -> str:
    """Extract assistant message text from an API response.
    Handles both 'content' and 'reasoning' fields (mlx_lm.server puts
    thinking-mode output in 'reasoning')."""
    if "error" in response:
        return ""
    try:
        msg = response["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or ""
        return content if content else reasoning
    except (KeyError, IndexError):
        return ""


def extract_code(text: str) -> str:
    """Extract Python code from a response, stripping think blocks and markdown."""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

    blocks = re.findall(r"```(?:python)?\s*\n([\s\S]*?)```", text)
    if blocks:
        return blocks[-1].strip()

    lines = text.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if re.match(r"^(def |class |import |from )", line):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip()

    return text.strip()


def get_usage(response: dict) -> dict:
    """Extract usage stats from API response."""
    usage = response.get("usage", {})
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }

# ---------------------------------------------------------------------------
# Test 1: Speed benchmark
# ---------------------------------------------------------------------------

def run_speed_benchmark(server_url: str) -> list:
    """Run speed benchmark prompts at temp=0."""
    print("=" * 60)
    print(" TEST 1: Speed Benchmark (temp=0)")
    print("=" * 60)

    results = []
    for prompt_cfg in SPEED_PROMPTS:
        name = prompt_cfg["name"]
        print(f"\n  [{name}] Sending request...", end=" ", flush=True)

        resp = chat_completion(
            server_url, prompt_cfg["messages"], temperature=0.0,
            max_tokens=prompt_cfg["max_tokens"],
        )

        elapsed = resp.get("_elapsed", resp.get("elapsed", 0))
        usage = get_usage(resp)
        comp_tokens = usage["completion_tokens"]
        tok_s = comp_tokens / elapsed if elapsed > 0 else 0

        result = {
            "name": name,
            "completion_tokens": comp_tokens,
            "prompt_tokens": usage["prompt_tokens"],
            "elapsed_s": round(elapsed, 2),
            "tok_s": round(tok_s, 1),
            "error": resp.get("error"),
        }
        results.append(result)

        if result["error"]:
            print(f"ERROR: {result['error']}")
        else:
            print(f"{comp_tokens} tokens in {elapsed:.1f}s = {tok_s:.1f} tok/s")

    valid = [r for r in results if not r.get("error")]
    if valid:
        avg_toks = sum(r["tok_s"] for r in valid) / len(valid)
        print(f"\n  Average: {avg_toks:.1f} tok/s across {len(valid)} prompts")

    return results

# ---------------------------------------------------------------------------
# Test 2: Quality pass-rate
# ---------------------------------------------------------------------------

def test_solution(code: str, test_code: str) -> bool:
    """Execute extracted code + tests in a subprocess."""
    full_code = code + "\n\n" + test_code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def run_quality_tests(server_url: str) -> list:
    """Run quality pass-rate tests at temp=0.7."""
    print("\n" + "=" * 60)
    print(f" TEST 2: Quality Pass-Rate (temp={QUALITY_TEMP}, N={QUALITY_SAMPLES})")
    print("=" * 60)

    results = []
    for problem in QUALITY_PROBLEMS:
        name = problem["name"]
        print(f"\n  [{name}]", flush=True)
        messages = [{"role": "user", "content": problem["prompt"]}]

        passes = 0
        sample_details = []
        for i in range(QUALITY_SAMPLES):
            print(f"    Sample {i+1}/{QUALITY_SAMPLES}...", end=" ", flush=True)

            resp = chat_completion(
                server_url, messages, QUALITY_TEMP, QUALITY_MAX_TOKENS
            )
            content = extract_content(resp)
            code = extract_code(content)
            passed = test_solution(code, problem["test_code"])

            if passed:
                passes += 1
                print("PASS")
            else:
                print("FAIL")

            usage = get_usage(resp)
            sample_details.append({
                "sample": i + 1,
                "passed": passed,
                "completion_tokens": usage["completion_tokens"],
                "code_length": len(code),
                "error": resp.get("error"),
            })

        rate = passes / QUALITY_SAMPLES
        result = {
            "name": name,
            "passes": passes,
            "total": QUALITY_SAMPLES,
            "pass_rate": round(rate, 2),
            "samples": sample_details,
        }
        results.append(result)
        print(f"    Result: {passes}/{QUALITY_SAMPLES} ({rate*100:.0f}%)")

    # Summary table
    print("\n  " + "-" * 50)
    print(f"  {'Problem':<35} {'Pass Rate':>10}")
    print("  " + "-" * 50)
    for r in results:
        bar = "#" * r["passes"] + "." * (r["total"] - r["passes"])
        print(f"  {r['name']:<35} {r['passes']:>2}/{r['total']} [{bar}]")
    overall = sum(r["passes"] for r in results)
    total = sum(r["total"] for r in results)
    print("  " + "-" * 50)
    print(f"  {'OVERALL':<35} {overall:>2}/{total} ({overall/total*100:.0f}%)")

    return results

# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------

def compare_results(file_a: str, file_b: str):
    """Print side-by-side comparison of two result files."""
    with open(file_a) as f:
        data_a = json.load(f)
    with open(file_b) as f:
        data_b = json.load(f)

    label_a = data_a.get("model_id", Path(file_a).stem)
    label_b = data_b.get("model_id", Path(file_b).stem)

    print("=" * 72)
    print(" COMPARISON")
    print(f"  A: {file_a}  ({label_a})")
    print(f"  B: {file_b}  ({label_b})")
    print("=" * 72)

    # Speed comparison
    print(f"\n{'SPEED BENCHMARK':^72}")
    print("-" * 72)
    print(f"  {'Prompt':<25} {'A tok/s':>10} {'B tok/s':>10} {'Delta':>10}")
    print("  " + "-" * 60)

    speed_a = {r["name"]: r for r in data_a.get("speed", [])}
    speed_b = {r["name"]: r for r in data_b.get("speed", [])}
    all_names = list(dict.fromkeys(list(speed_a) + list(speed_b)))

    for name in all_names:
        a_tok = speed_a.get(name, {}).get("tok_s", 0)
        b_tok = speed_b.get(name, {}).get("tok_s", 0)
        delta = b_tok - a_tok
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<25} {a_tok:>10.1f} {b_tok:>10.1f} {sign}{delta:>9.1f}")

    valid_a = [r["tok_s"] for r in data_a.get("speed", []) if not r.get("error")]
    valid_b = [r["tok_s"] for r in data_b.get("speed", []) if not r.get("error")]
    if valid_a and valid_b:
        avg_a = sum(valid_a) / len(valid_a)
        avg_b = sum(valid_b) / len(valid_b)
        delta = avg_b - avg_a
        sign = "+" if delta >= 0 else ""
        print("  " + "-" * 60)
        print(f"  {'AVERAGE':<25} {avg_a:>10.1f} {avg_b:>10.1f} {sign}{delta:>9.1f}")

    # Quality comparison
    print(f"\n{'QUALITY PASS-RATE':^72}")
    print("-" * 72)
    print(f"  {'Problem':<30} {'A':>8} {'B':>8} {'Delta':>8}")
    print("  " + "-" * 58)

    qual_a = {r["name"]: r for r in data_a.get("quality", [])}
    qual_b = {r["name"]: r for r in data_b.get("quality", [])}
    all_qnames = list(dict.fromkeys(list(qual_a) + list(qual_b)))

    total_a = total_b = 0
    count_a = count_b = 0
    for name in all_qnames:
        ra = qual_a.get(name, {})
        rb = qual_b.get(name, {})
        a_rate = ra.get("pass_rate", 0)
        b_rate = rb.get("pass_rate", 0)
        a_str = f"{ra.get('passes',0)}/{ra.get('total',0)}"
        b_str = f"{rb.get('passes',0)}/{rb.get('total',0)}"
        delta = b_rate - a_rate
        sign = "+" if delta >= 0 else ""
        print(f"  {name:<30} {a_str:>8} {b_str:>8} {sign}{delta*100:>6.0f}%")
        total_a += ra.get("passes", 0)
        total_b += rb.get("passes", 0)
        count_a += ra.get("total", 0)
        count_b += rb.get("total", 0)

    print("  " + "-" * 58)
    if count_a and count_b:
        oa = total_a / count_a
        ob = total_b / count_b
        delta = ob - oa
        sign = "+" if delta >= 0 else ""
        print(f"  {'OVERALL':<30} {total_a}/{count_a:>4}  {total_b}/{count_b:>4}  {sign}{delta*100:>6.0f}%")

    # Verdict
    print("\n" + "=" * 72)
    if count_b and count_a:
        if ob > oa + 0.05:
            print("  VERDICT: B shows meaningful quality improvement.")
        elif ob < oa - 0.05:
            print("  VERDICT: B shows quality regression.")
        else:
            print("  VERDICT: Quality is within noise (delta < 5%).")
    print("=" * 72)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test merged model vs base")
    parser.add_argument("--server", help="Server URL")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--model-id", default="unknown", help="Model identifier")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="Compare two result JSONs")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    if not args.server or not args.output:
        parser.error("--server and --output required for benchmark mode")

    global _MODEL_ID
    _MODEL_ID = args.model_id

    speed_results = run_speed_benchmark(args.server)
    quality_results = run_quality_tests(args.server)

    output = {
        "model_id": args.model_id,
        "server_url": args.server,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "speed": speed_results,
        "quality": quality_results,
        "summary": {
            "avg_tok_s": round(
                sum(r["tok_s"] for r in speed_results if not r.get("error"))
                / max(1, len([r for r in speed_results if not r.get("error")])),
                1,
            ),
            "overall_pass_rate": round(
                sum(r["passes"] for r in quality_results)
                / max(1, sum(r["total"] for r in quality_results)),
                3,
            ),
        },
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
