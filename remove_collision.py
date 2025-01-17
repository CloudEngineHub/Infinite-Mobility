import xml.etree.ElementTree as ET
import os
path = '/home/pjlab/datasets/partnet_mobility'

def remove_collision(path):
    print(path)
    tree = ET.parse(path)
    root = tree.getroot()
    ls = root.findall('link')
    for l in ls:
        for c in l.findall('./collision'):
            print(c)
            l.remove(c)
    tree.write(path.replace('.urdf', '_no_collision.urdf'))
    return tree


def iter_path(path):
    print(path)
    fs = os.listdir(path)
    for f in fs:
        if os.path.isdir(f"{path}/{f}"):
            iter_path(f"{path}/{f}")
        else:
            if f.endswith('urdf'):
                remove_collision(f"{path}/{f}")
iter_path(path)


#tree.write("updated_bookstore_data.xml")

#print("XML file has been successfully updated.")
