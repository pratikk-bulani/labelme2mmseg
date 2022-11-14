import json, glob, os, numpy as np, cv2, argparse

def parse_args():
    parser = argparse.ArgumentParser(prog = 'labelme2mmseg', description = 'Converts LabelMe JSON to mmsegmentation segmentation mask')
    parser.add_argument('-o', '--output_dir', default="../dataset_mmseg", type=str)
    parser.add_argument('-i', '--input_dir', default="./", type=str)
    parser.add_argument('-l', '--label_file', default="./mg_hector_labels.txt", type=str)
    args = parser.parse_args()
    return args
args = parse_args()

os.makedirs(args.output_dir, exist_ok=True)

with open(args.label_file) as f:
    labels_actual = {l.replace("\n", ""):i for i, l in enumerate(f.readlines())}

for json_path in glob.glob(os.path.join(args.input_dir, "*.json")):
    with open(json_path, 'r') as f:
        json_content = json.load(f)
    seg_mask = np.zeros(shape=(json_content['imageHeight'], json_content['imageWidth']), dtype=np.uint8)
    fill_mask_order = [None] * len(labels_actual)
    for shape in json_content['shapes']:
        if shape['label'] not in labels_actual:
            print("Error:", os.path.basename(json_path), shape['label'])
        else:
            class_label = labels_actual[shape['label']]
            fill_mask_order[class_label] = np.rint(np.array(shape['points']), dtype='int32')
    for i in fill_mask_order:
        cv2.fillPoly(seg_mask, pts=[i], color=class_label)
    cv2.imwrite(os.path.join(args.output_dir, os.path.splitext(os.path.basename(json_path))[0] + ".png"), seg_mask)