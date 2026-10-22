import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
log_file = r"C:\Users\LUU VAN THANH HUY\.gemini\antigravity\brain\ffa968d4-2dfd-49f5-90a4-081c68844b5b\.system_generated\logs\transcript_full.jsonl"
try:
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if 'yolo11m_p2_visdrone-4' in line and 'yolo val' in line:
                try:
                    data = json.loads(line)
                    if data.get("type") == "PLANNER_RESPONSE":
                        tool_calls = data.get("tool_calls", [])
                        for tc in tool_calls:
                            cmd = tc.get("args", {}).get("CommandLine", "")
                            if 'yolo val' in cmd:
                                print("P2 Command:", cmd)
                except Exception:
                    pass
except Exception as e:
    pass
