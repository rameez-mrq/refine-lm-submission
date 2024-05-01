import json

import _pickle as pickle
import argparse


def load_input(path):
    rs = []
    with open(path, 'r') as f:
        json_data = json.load(f)
        for key, ex in json_data.items():
            context = ex['context'].strip()
            choices = [ex['q0']['ans0']['text'].strip(), ex['q0']
            ['ans1']['text'].strip()]
            questions = [ex['q0']['question'].strip(), ex['q1']
            ['question'].strip()]
            subj0_cluster, subj1_cluster, subj0, subj1, tid, a_cluster, obj0, obj1 = key.strip().split('|')
            rs.append(((subj0_cluster, subj1_cluster), (subj0, subj1),
                       tid, a_cluster, (obj0, obj1), context, choices, questions))
    return rs


def preprocess(source, prompt=None):
    def create_sentence(context, q, choices):
        return prompt.replace("[ans0]", choices[0]).replace("[ans1]", choices[1]).replace("[context]", context).replace(
            "[question]", q)

    rs = []
    for i, (scluster, spair, tid, acluster, opair, context, choices, questions) in enumerate(source):
        for j, q in enumerate(questions):

            if prompt is None:
                rs.append(((i, j), scluster, spair, tid, acluster,
                           opair, context + ' ' + q, choices))
            else:
                rs.append(((i, j), scluster, spair, tid, acluster,
                           opair, create_sentence(context, q, choices), choices))
    return rs


def pairwise(i):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(i)
    return zip(a, a)


def create_pickle(preprocessed):
    new_pkl = {}
    for a, b in pairwise(preprocessed):
        if ((a[2][1], a[2][0]), a[5], a[3]) in new_pkl.keys():
            new_pkl[((a[2][1], a[2][0]), a[5], a[3])
            ].extend([list(a), list(b)])
        else:
            k = (a[2], a[5], a[3])
            new_pkl[k] = [list(a), list(b)]
    return new_pkl


print("Starting")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_path", help='The path of the input json file', required=True)

parser.add_argument(
    "--output", help='The name of the input pkl file', required=True)
parser.add_argument(
    "--prompt", help='Template for the LLM prompt', required=False, default=None)
args = parser.parse_args()

input_path = args.input_path
output = args.output
rs = load_input(input_path)
pprs = preprocess(rs, args.prompt)
new_pkl = create_pickle(pprs)
with open(output, "wb") as f:
    pickle.dump(new_pkl, f)
    f.close()
print("done")