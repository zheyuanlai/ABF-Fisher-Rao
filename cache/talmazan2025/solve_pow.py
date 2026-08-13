import hashlib, re, subprocess, sys

UA = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
URL = "https://pmc.ncbi.nlm.nih.gov/articles/instance/13284794/bin/NIHMS2186658-supplement-tutorial_files.zip"

# fresh interstitial (challenge may rotate)
html = subprocess.run(["curl","-s","-A",UA,"-c","cookies.txt",URL], capture_output=True, text=True).stdout
m = re.search(r'POW_CHALLENGE = "([^"]+)"', html)
if not m:
    print("no challenge; maybe got file directly?"); sys.exit(1)
chal = m.group(1)
diff = int(re.search(r'POW_DIFFICULTY = "(\d+)"', html).group(1))
print("challenge:", chal, "difficulty:", diff)

nonce = 0
prefix = "0"*diff
while True:
    h = hashlib.sha256((chal+str(nonce)).encode()).hexdigest()
    if h.startswith(prefix):
        break
    nonce += 1
print("nonce:", nonce, "hash:", h)
cookie = f"{chal},{nonce}"
r = subprocess.run(["curl","-s","-A",UA,"-b",f"cloudpmc-viewer-pow={cookie}","-c","cookies.txt",
                    "-o","tutorial_files.zip","-w","%{http_code} %{size_download}",URL],
                   capture_output=True, text=True)
print("download:", r.stdout)
