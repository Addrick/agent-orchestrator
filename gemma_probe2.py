import time
from eval_harnesses.suites.memory_recall.lme_judge import GeminiACPClient, _gemini_cwd
cwd=_gemini_cwd()
for m in ["gemma-4-31b-it","gemma-4-26b-a4b-it"]:
    for attempt in (1,2):
        try:
            c=GeminiACPClient(m, cwd)
            r=c.ask("Reply with exactly one token: yes")
            print(f"OK   {m:<22} (try {attempt}) -> {r[:60]!r}")
            c.close(); break
        except Exception as e:
            print(f"FAIL {m:<22} (try {attempt}) -> {type(e).__name__}: {str(e)[:150]}")
            time.sleep(2)
