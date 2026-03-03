import json
import sys
from pathlib import Path

REQUIRED_TOP_LEVEL = ["implemented_features", "qa", "demo"]

def fail(msg: str):
    print(f"VERIFY_FAIL: {msg}")
    sys.exit(1)

def is_non_empty_str(x) -> bool:
    return isinstance(x, str) and len(x.strip()) > 0

def main():
    if len(sys.argv) != 2:
        fail("Usage: verify_output.py <artifacts/sanity_output.json>")

    path = Path(sys.argv[1])
    if not path.exists():
        fail(f"File not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        fail(f"Invalid JSON: {e}")

    for k in REQUIRED_TOP_LEVEL:
        if k not in data:
            fail(f"Missing top-level key: {k}")

    feats = data.get("implemented_features")
    if not isinstance(feats, list):
        fail("implemented_features must be a list like ['A','B']")

    feat_set = set(feats)

    qa = data.get("qa")
    if not isinstance(qa, list):
        fail("qa must be a list")

    demo = data.get("demo")
    if not isinstance(demo, dict):
        fail("demo must be an object")

    if "A" in feat_set:
        if len(qa) == 0:
            fail("Feature A claimed but qa is empty")

        for i, item in enumerate(qa):
            if not isinstance(item, dict):
                fail(f"qa[{i}] must be an object")

            if not is_non_empty_str(item.get("question")):
                fail(f"qa[{i}].question missing/empty")

            if not is_non_empty_str(item.get("answer")):
                fail(f"qa[{i}].answer missing/empty")

            citations = item.get("citations")
            if not isinstance(citations, list) or len(citations) == 0:
                fail(f"qa[{i}] must include non-empty citations[]")

            for j, c in enumerate(citations):
                if not isinstance(c, dict):
                    fail(f"qa[{i}].citations[{j}] must be an object")
                if not is_non_empty_str(c.get("source")):
                    fail(f"qa[{i}].citations[{j}].source missing/empty")
                if not is_non_empty_str(c.get("locator")):
                    fail(f"qa[{i}].citations[{j}].locator missing/empty (page/section/chunk id)")
                if not is_non_empty_str(c.get("snippet")):
                    fail(f"qa[{i}].citations[{j}].snippet missing/empty")

    if "B" in feat_set:
        user_mem = Path("USER_MEMORY.md")
        comp_mem = Path("COMPANY_MEMORY.md")
        if not user_mem.exists() or not comp_mem.exists():
            fail("Feature B claimed but USER_MEMORY.md / COMPANY_MEMORY.md not found")

        mem_writes = demo.get("memory_writes")
        if not isinstance(mem_writes, list) or len(mem_writes) == 0:
            fail("Feature B claimed but demo.memory_writes is empty")

        for i, w in enumerate(mem_writes):
            if not isinstance(w, dict):
                fail(f"demo.memory_writes[{i}] must be an object")
            if w.get("target") not in ("USER", "COMPANY"):
                fail(f"demo.memory_writes[{i}].target must be 'USER' or 'COMPANY'")
            if not is_non_empty_str(w.get("summary")):
                fail(f"demo.memory_writes[{i}].summary missing/empty")

    print("VERIFY_OK")

if __name__ == "__main__":
    main()