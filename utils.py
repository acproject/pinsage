import torch
import dgl
numSymps = 360
numHerbs = 760
# d = {}
# with open(r'C:\Users\wxy\Desktop\KGAT-wxy\knowledge_graph_attention_network-master\Data\TCM_small\train.txt', 'r') as f:
#     for line in f.readlines():
#         temp = line.strip().split()
#         herbs = temp[1:]
#         symp = temp[0]
#         herbs = ' '.join(herbs)
#         if herbs not in d:
#             d[herbs] = [symp]
#         else:
#             d[herbs].append(symp)
#
# with open('train.txt', 'w') as f:
#     for k, v in d.items():
#         f.write(' '.join(v) + '\t' + k + '\n')

def getD(file):
    d = {}
    with open(file, 'r', encoding='utf8') as f:
        for line in f.readlines()[1:]:
            temp = line.strip().split()
            d[temp[0]] = temp[1]
    return d
#
# sympsD = getD(r'C:\Users\wxy\Desktop\KGAT-wxy\knowledge_graph_attention_network-master\Data\TCM_small\user_list.txt')
# herbsD = getD(r'C:\Users\wxy\Desktop\KGAT-wxy\knowledge_graph_attention_network-master\Data\TCM_small\herb_list.txt')
#
# with open(r'C:\Users\wxy\Desktop\RippleNet-PyTorch\data\TCM\OG\herbs_test_og.txt', 'r', encoding='utf8') as f, \
#         open(r'C:\Users\wxy\Desktop\RippleNet-PyTorch\data\TCM\OG\symps_test_og.txt', 'r', encoding='utf8') as w,\
#             open(r'test.txt', 'w') as o:
#     herbs = f.readlines()
#     symps = w.readlines()
#     for line in range(len(herbs)):
#         herb = [herbsD[i] for i in herbs[line].strip().split()]
#         symp = [sympsD[i] for i in symps[line].strip().split()]
#         o.write(' '.join(symp) + '\t' + ' '.join(herb) + '\n')

def load_data(filePath):
    print('Loading data from '+filePath)
    g = dgl.DGLGraph()
    edges = []
    symps = []
    herbs = []
    # todo notice 读入的文件本身是str类型，需要转成int
    with open(filePath, 'r', encoding='utf8') as f:
        for line in f.readlines():
            temp = line.strip().split('\t')
            symps = [int(i) for i in temp[0].split()]
            herbs = [int(i) for i in temp[1].split()]
            for s in symps:
                for h in herbs:
                    edges.append((s, h))

    label_sh = torch.zeros(numSymps, numHerbs, dtype=torch.float)

    for i in edges:
        label_sh[i[0]][i[1]] = 1

    print('Load finished')
    return label_sh

with open(r'D:\pinsage\data\train.txt', 'r', encoding='utf8') as f, \
    open('data/edge.txt', 'w') as o:
    for line in f.readlines():
        temp = line.strip().split('\t')
        symps = temp[0].strip().split()
        herbs = temp[1].strip().split()
        for symp in symps:
            for herb in herbs:
                o.write(symp + ' ' + herb + '\n')


