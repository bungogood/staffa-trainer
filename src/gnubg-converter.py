import codecs
import sys

def to_posid(pos: str) -> str:
    pid = "".join(hex(ord(c) - ord("A"))[2:] for c in pos)
    b64 = codecs.encode(codecs.decode(pid, 'hex'), 'base64').decode().strip()
    return b64[:-2]

def convert(inpath: str) -> None:
    outpath = filepath.replace(".txt", ".csv")
    output = []
    with open(inpath, 'r') as f:
        positions = f.readlines()
        for line in positions:
            if line.startswith('#'):
                continue
            line = line.strip()
            line = line.split(' ')
            line[0] = to_posid(line[0])
            output.append(",".join(line))
    with open(outpath, 'w') as f:
        f.write("positionid,win,wing,winbg,loseg,losebg\n")
        f.write("\n".join(output))
        f.write("\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gnubg-converter.py <gnu-train-data.txt>")
        sys.exit(1)
    
    for filepath in sys.argv[1:]:
        convert(filepath)
