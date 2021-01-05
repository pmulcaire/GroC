import sys
from collections import defaultdict
import IPython as ipy


def read_text(filename, length):
    lines = []
    for line in open(filename,'r'):
        lines.append(line.strip())
        if len(lines) > length:
            break
    return lines


def simplify(lines):
    punct = [chr(45),chr(8211),chr(8212)]
    newlines = []
    for line in lines:
        line = line.lower()
        newline = []
        for w in line.split():
            if w in punct:
                newline.append(w)
            elif w[0]  in punct:
                newline.append(w[0])
                newline.append(w[1:])
            elif w[-1] in punct:
                newline.append(w[:-1])
                newline.append(w[-1])
            else:
                for p in punct:
                    splits = w.split(p)
                    w = ' '.join([s+' '+p for s in splits[:-1]] + [splits[-1]])
                newline.append(w)
        newlines.append(' '.join(newline))
    return newlines


def vocab_count(lines):
    v = defaultdict(int)
    for line in lines:
        for w in line.split():
            v[w] += 1
    return len(v)


if __name__=="__main__":
    text = read_text(sys.argv[1], int(sys.argv[2]))
    new_text = simplify(text)
    print("Old vocab: {} New vocab: {}".format(vocab_count(text),
                                               vocab_count(new_text)),
          file=sys.stderr)
    for line in new_text:
        print(line, flush=True)
