import numpy as np
import sys

def extract_tokens_and_time(st):
    arrs = st.strip().split()
    num_tokens = int(arrs[10][:-1])
    time = float(arrs[-1])
    return num_tokens, time

assert len(sys.argv) == 4, "Usage: python {sys.argv[0]} <log_file_path_prefix> <prefill/decode> <nrun>"
file_name_prefix = sys.argv[1]
target = sys.argv[2]
nrun = int(sys.argv[3])
assert target in ["prefill", "decode"]


prefill_data = []
decode_data = []
for i in range(nrun+1):
    try:
        with open(file_name_prefix+f"_run{i}") as f:
            lines = f.readlines()
            # treat first 10 steps as warmup
            prefill_data += [extract_tokens_and_time(x) for x in lines if "ccc ||" in x][10:]
            decode_data += [extract_tokens_and_time(x) for x in lines if "kkk ||" in x][10:]
    except Exception as e:
        #print(e)
        continue
assert len(prefill_data) > 1 and len(decode_data) > 1, f"{len(prefill_data), len(decode_data)}"
prefill_fit = np.polyfit([x[0] for x in prefill_data] , [x[1] for x in prefill_data], 1)
decode_fit = np.polyfit([x[0] for x in decode_data] , [x[1] for x in decode_data], 1)
if target == "prefill":
    print(prefill_fit[0], prefill_fit[1])
else:
    print(decode_fit[0], decode_fit[1])

    
    
