import os

def add_padding(C__,input_file,output_file):
    with open(input_file,'r') as f, open(output_file, "w") as out_f:
        for line in f:
            l = len(line.strip().split(' '))
            padding = "<blank> "*(C__.PADDING_TOKEN - l -1)
            outline ="<s> " + " ".join(line.strip().split(' ')) + " </s> "+padding+"\n"
            out_f.write(outline)

def padding_datasets(raw_path, padded_path):
    for data in os.listdir(raw_path):
        filename,data_type = data.split('.')
        padded_item = os.path.join(padded_path,filename+"_padded."+data_type)
        raw_item = os.path.join(raw_path,data)
        add_padding(raw_item,padded_item)

