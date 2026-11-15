#!/usr/bin/env python3
"""Download verified dataset release parts; optionally extract the original tree."""
from pathlib import Path
import argparse,json,hashlib,urllib.request,subprocess,re,sys,os
HERE=Path(__file__).resolve().parent

def digest(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",type=Path,default=HERE);p.add_argument("--extract",action="store_true");a=p.parse_args();out=a.output_dir.expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
 meta=json.loads((HERE/"dataset_release_manifest.json").read_text());assert meta["repo"]=="ShenShan123/R2G";parts=[];base="https://github.com/"+meta["repo"]+"/releases/download/"+meta["release_tag"]+"/"
 for chunk in meta["parts"]:
  name=chunk["name"];assert re.fullmatch(r"R2G_multiview_v3_dataset[.]tar[.]zst[.]part-[0-9]{3}",name)
  target=out/name
  if target.exists() and target.stat().st_size==chunk["bytes"] and digest(target)==chunk["sha256"]:print("Verified existing",name,flush=True)
  else:
   temp=target.with_suffix(target.suffix+".partial");print("Downloading",name,flush=True)
   req=urllib.request.Request(base+name,headers={"User-Agent":"R2G300-dataset-downloader"})
   with urllib.request.urlopen(req,timeout=120) as response,temp.open("wb") as f:
    while True:
     b=response.read(8*1024**2)
     if not b:break
     f.write(b)
   assert temp.stat().st_size==chunk["bytes"] and digest(temp)==chunk["sha256"],name+" checksum mismatch"
   os.replace(temp,target)
  parts.append(target)
 h=hashlib.sha256()
 for target in parts:
  with target.open("rb") as f:
   for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 assert h.hexdigest()==meta["archive_sha256"],"Combined archive mismatch"
 if a.extract:
  if (out/"R2G_multiview_v3_dataset").exists():p.error("Dataset directory already exists; use a new output directory.")
  cat=subprocess.Popen(["cat"]+list(map(str,parts)),stdout=subprocess.PIPE)
  decoder=subprocess.Popen(["zstd","-d","-c"],stdin=cat.stdout,stdout=subprocess.PIPE);cat.stdout.close()
  tar=subprocess.Popen(["tar","-xf","-","-C",str(out)],stdin=decoder.stdout);decoder.stdout.close()
  codes=[tar.wait(),decoder.wait(),cat.wait()];assert codes==[0,0,0],codes
  assert (out/"R2G_multiview_v3_dataset/manifest.csv").is_file()
 print("Dataset download and checksums completed.")
if __name__=="__main__":main()
