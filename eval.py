import os
import sys
dir_name = os.path.abspath(os.path.dirname("__name__"))
sys.path.append(dir_name)
from nlgeval import NLGEval
import json
import jieba
from metrics.dist import dist_compute


def refer_process(refer_str_list):
    result_list = []
    for r in refer_str_list:
        r_list = r.split("#")   #pre tokenize
        n_r = " ".join(r_list)
        result_list.append(n_r)
    return result_list


def gen_process(gen_str):
    result_str = list(jieba.cut(gen_str))
    result_str = " ".join(result_str)
    return result_str


def score(generated_path, refer_path):
    with open(generated_path, "r", encoding="utf-8") as f:
        generated_dict = json.load(f)

    with open(refer_path, "r", encoding="utf-8") as g:
        refer_dict = json.load(g)

    refer_num = len(list(refer_dict.values())[0])
    print("references num: ", refer_num)

    gen_eval_list = []
    ref_eval_list = []

    for id_p, g in generated_dict.items():
        gen_eval_list.append(gen_process(g))
        refer_tmp_list = refer_dict[id_p]
        assert len(refer_tmp_list) == refer_num
        ref_eval_list.append(refer_process(refer_tmp_list))

    ref_eval_list = list(zip(*ref_eval_list))

    nlgeval = NLGEval()

    metric_dict = nlgeval.compute_metrics(ref_eval_list, gen_eval_list)
    print(metric_dict)
    
    dist_dict = dist_compute(gen_eval_list, [1, 2])
    print(dist_dict)


if __name__ == "__main__":
    gen_path = ".../generated_text.json"
    ref_path = ".../references.json"
    print("generate model dir name: ", gen_path)
    score(gen_path, ref_path)


