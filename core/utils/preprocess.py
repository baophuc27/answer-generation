import os
import json
# def add_padding(__C,input_file,output_file):
#     with open(input_file,'r') as f, open(output_file, "w") as out_f:
#         for line in f:
#             l = len(line.strip().split(' '))
#             padding = "<PAD> "*(__C.PADDING_TOKEN - l -1)
#             outline ="<EOS> " + " ".join(line.strip().split(' ')) + " <SOS> "+padding+"\n"
#             out_f.write(outline)

# def padding_datasets(__C,raw_path, padded_path):
#     for data in os.listdir(raw_path):
#         filename,data_type = data.split('.')
#         padded_item = os.path.join(padded_path,filename+"_padded."+data_type)
#         raw_item = os.path.join(raw_path,data)
#         add_padding(__C,raw_item,padded_item)

TYPE_MAPPING = {
    'ans':'answers',
    'ques':'questions',
    'tgt':'targets'
}

def preprocess(raw_path,processed_path):
    for data in os.listdir(raw_path):
        filename,data_type = data.split('.')
        processed_item = os.path.join(processed_path,filename+"_"+data_type+".json")
        raw_item = os.path.join(raw_path,data)
        convert_to_json(raw_item,processed_item,TYPE_MAPPING[data_type])

def convert_to_json(input_file,output_file, data_type):
    data = {}
    data[data_type] =[]
    with open(input_file,'r') as f, open(output_file, "w") as out_f:
        for ix,line in enumerate(f):
            data[data_type].append({ix: line[:-2]})
        json.dump(data,out_f)

