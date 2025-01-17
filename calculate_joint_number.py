import xml.etree.ElementTree as ET
import os
import json
path = '/home/pjlab/datasets/partnet_mobility'

catagory = 'chair'
json_path = f'./partnet_catagory_json/{catagory}.json'
res = json.load(open(json_path))
objs = [o['id'] for o in res]
number = 0
sum = 0
for o in objs:
    print(o)
    p = f"{path}/{o}/mobility.urdf"
    tree = ET.parse(p)
    root = tree.getroot()
    js = root.findall('joint')
    for j in js:
        if j.attrib['type'] == 'fixed':
            print(j)
            sum -= 1
    sum += len(js)
    print(len(js))
    number += 1
    #print(js)
print(sum/number)