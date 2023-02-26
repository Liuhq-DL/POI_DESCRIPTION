

def process_list(input_list):
    remove_list = ['"', "'"]
    out_list = []
    for a in input_list:
        b = a.strip()
        for i in remove_list:
            b = b.replace(i, "")
        out_list.append(b)
    return out_list
