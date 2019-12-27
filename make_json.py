import json
import gzip

def parse(path):
	g = gzip.open(path, 'r')
	for l in g:
		yield json.dumps(eval(l))

f = open("data/meta.json", 'w')
i = 1
for l in parse("data/meta_Clothing_Shoes_and_Jewelry.json.gz"):
	i = i + 1
	print(i)
	f.write(l + '\n')