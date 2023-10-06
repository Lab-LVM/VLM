import ujson

name = '/data/sampling/soonge_imageNet_16s.json'

with open(name) as f:
    data = ujson.load(f)

for i in range(len(data['s_imgs'])):
    data['s_imgs'][i] = data['s_imgs'][i].replace('/vlm', '')

with open(name, 'w') as f:
    ujson.dump(data, f, indent=4, ensure_ascii=False)
