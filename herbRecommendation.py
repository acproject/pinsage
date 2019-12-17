import os
import dgl
import torch
import pandas as pd
from collections import defaultdict

class HerbRecommendation:
    def __init__(self, directory):
        '''
        directory: path to movielens directory which should have the three
                   files:
                   users.dat
                   movies.dat
                   ratings.dat
        '''
        self.directory = directory
        symps = []
        herbs = []
        ratings = []

        for s in range(360):
            for h in range(760):
                ratings.append({
                    'symp_id': s,
                    'herb_id': h,
                    'rating': 0,
                })

        with open(os.path.join(directory, 'herb_list.txt'), encoding='utf8') as f:
            for line in f.readlines():
                herb_name, id_ = line.strip().split()
                herbs.append({'id': int(id_),
                              'herb_name': herb_name
                              })
        self.herbs = pd.DataFrame(herbs).set_index('id')

        with open(os.path.join(directory, 'user_list.txt'), encoding='utf8') as f:
            for line in f.readlines():
                symp_name, id_ = line.strip().split()
                symps.append({'id': int(id_),
                              'symp_name': symp_name
                              })
        self.symps = pd.DataFrame(symps).set_index('id')

        with open(os.path.join(directory, 'edge.txt'), encoding='utf8') as f:
            for line in f.readlines():
                symp, herb = line.strip().split()
                ratings.append({
                    'symp_id': int(symp),
                    'herb_id': int(herb),
                    'rating': 1,
                })
        self.ratings = pd.DataFrame(ratings)

        # test_set = self.ratings[22917:25917].index
        # valid_set = self.ratings[25917:].index
        # self.ratings['valid'] = self.ratings.index.isin(valid_set)
        # self.ratings['test'] = self.ratings.index.isin(test_set)


    def todglgraph(self):
        '''
        returns:
        g, user_ids, movie_ids:
            The DGL graph itself.  Each edge has a binary feature "valid" and a binary
            feature "test" indicating validation/test example.
            The list of user IDs (node i corresponds to user user_ids[i])
            The list of movie IDs (node i + len(user_ids) corresponds to movie movie_ids[i])
        '''
        symp_ids = list(self.symps.index)
        herb_ids = list(self.herbs.index)

        symp_ids_invmap = {id_: i for i, id_ in enumerate(symp_ids)}
        herb_ids_invmap = {id_: i for i, id_ in enumerate(herb_ids)}

        g = dgl.DGLGraph()
        g.add_nodes(len(symp_ids) + len(herb_ids))
        rating_user_vertices = [symp_ids_invmap[id_] for id_ in self.ratings['symp_id'].values]
        rating_movie_vertices = [herb_ids_invmap[id_] + len(symp_ids)
                                 for id_ in self.ratings['herb_id'].values]

        g.add_edges(rating_user_vertices, rating_movie_vertices)
        g.add_edges(rating_movie_vertices, rating_user_vertices)

        return g, symp_ids, herb_ids

hr = HerbRecommendation('data')
g, symp_ids, herb_ids = hr.todglgraph()