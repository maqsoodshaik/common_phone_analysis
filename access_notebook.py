# -*- coding: utf-8 -*-
"""access_notebook.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1i6kOJf_4cQTXUAlC1QXwYWzXesX8JWVa

imports
"""

import os
import pickle
import sim
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import random
import seaborn as sns
import numpy as np
from collections import Counter

"""functions to load the saved file which contains mapping from phoneme to codebook entry"""

def load_ph_code_entry_map(
    path_phn_file: str#path of file
):  # loading phoneme to codeentry mapping for the file
    with open(path_phn_file, "rb") as f:
        phn_dict = pickle.load(f)
    return phn_dict


def load_ph_code_entry_map_folder(
    pickle_path#path of folder
):  # loading phoneme to codeentry mapping for the whole folder
    phn_dict = {}
    for subdir, dirs, files in os.walk(pickle_path):
        for index,file in enumerate(files):
            if index <= 5000 :
                if ".pkl" in file:
                    phn_dict_f = load_ph_code_entry_map(
                        subdir + "/" + file
                    )  # loading phoneme to codeentry mapping for the file
                    for p, val in phn_dict_f.items():
                        phn_dict_tmp = dict(Counter(val))
                        if p in phn_dict:
                            for dis,count in phn_dict_tmp.items():
                                if dis in phn_dict[p]:
                                    phn_dict[p][dis]+=count
                                else:
                                    phn_dict[p][dis]=count
                        else:
                            phn_dict[p] = phn_dict_tmp
    return phn_dict

"""
function to plot compression"""

def plot_compress(sim_mt, phn_to_dist_1_keys,type_of_compression = "TSNE", kmeans = False):
    if type_of_compression == "mds":
        compress = MDS(n_components=2, dissimilarity="precomputed", n_jobs=-1).fit_transform(
            sim_mt
        )
    elif type_of_compression == "TSNE":
        # compress = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3,random_state = 0).fit_transform(sim_mt)
        compress = TSNE(n_components=2,verbose=1, perplexity=9, n_iter=1000, learning_rate=200,random_state = 0).fit_transform(sim_mt)
    elif type_of_compression == "PCA":
        compress = PCA(n_components=2,random_state = 0).fit_transform(sim_mt)  
        # val1 = compress[:,0]
        # val2 = compress[:,1]
        # compress[:,1] = np.array(val1)*-1
        # compress[:,0] = np.array(val2)*-1
    if kmeans == True:
        kmns = KMeans(n_clusters=12, random_state=0).fit(compress)
        colors_cluster = kmns.labels_
        sns.scatterplot(compress[:, 0], compress[:, 1],c = colors_cluster)
    else:
        sns.scatterplot(compress[:, 0], compress[:, 1])
    for i, phn in enumerate(phn_to_dist_1_keys):
        v = random.uniform(0, 0.5)
        plt.annotate(phn, (compress[:, 0][i], compress[:, 1][i]))
    plt.xlabel("{type_of_compression} axis 1")
    plt.ylabel("{type_of_compression} axis 2")
    plt.savefig("compress.pdf", bbox_inches="tight")

"""providing path folder which contains the phoneme to code entry mapping"""

# codebook = 1
# folder_name = "timit_pkl_xlsr"
# path_folder = (
#     f"/Users/mohammedmaqsoodshaik/Desktop/hiwi/task1/{folder_name}/codebook{codebook}/"
# )
# phn_dict1 = load_ph_code_entry_map_folder(path_folder)
# codebook = 2
# folder_name = "timit_pkl_xlsr"
# path_folder = (
#     f"/Users/mohammedmaqsoodshaik/Desktop/hiwi/task1/{folder_name}/codebook{codebook}/"
# )
# phn_dict2 = load_ph_code_entry_map_folder(path_folder)
# for key_1,val_1 in phn_dict1.items():
#         phn_dict1[key_1] +=  list(np.asarray(phn_dict2[key_1]) + 320)
# phn_dict_final_xlsr=phn_dict1
codebook = 1
folder_name = "CP_wav2vec2_pkl"
lang = "en"
path_folder = (
    f"/data/corpora/common_phone_analysis/{folder_name}/codebook_{codebook}/{lang}"
)
phn_dict1 = load_ph_code_entry_map_folder(path_folder)
codebook = 2
folder_name = "CP_wav2vec2_pkl"
path_folder = (
    f"/data/corpora/common_phone_analysis/{folder_name}/codebook_{codebook}/{lang}"
)
phn_dict2 = load_ph_code_entry_map_folder(path_folder)
for key_1,val_1 in phn_dict2.items():
        for dis in val_1:
            phn_dict1[key_1][dis+320] = phn_dict2[key_1][dis]
phn_dict_final_wav2vec2=phn_dict1

#RU phonemes
# ['??',
#  'i',
#  'e',
#  'o',
#  'u',
#  'a',
#  'l',
#  'j',
#  'm',
#  'n','m??', 'n??',
#  'b',
#  'd',
#  'p','p??',
#  't',
#  'k','??', '????',
#  'b??', 'd??','k??','ts', 't????', 't??', 'v??','z??',
#  'f??',
#  'v','x','x??',
#  'z',
#  'f',
#  's','s??','????', '??',
#  '??','l??','r', 'r??'
#  ]

def adding_missing_phn_count(set_of_val, phn_dict, phn_name):
    phn_map_counts = phn_dict[phn_name]

    temp_lst = []
    # creating list of counts of code entries even including the codeentries which are not present
    temp_lst = [
        phn_map_counts[val] if val in phn_map_counts else 0 for val in set_of_val
    ]
    temp_lst = [float(i) / sum(temp_lst) for i in temp_lst]
    return temp_lst
def smoothing_dist(phn_dict, set_of_val, abs_discount):
    phn_to_dist = {}
    for phn_name in phn_dict:
        temp_lst = adding_missing_phn_count(set_of_val, phn_dict, phn_name)
        temp_lst_smoothed = sim.absolute_discounting(temp_lst, abs_discount, set_of_val)
        phn_to_dist[phn_name] = temp_lst_smoothed
    return phn_to_dist

def similarity_calculation(phn_to_dist_1, phn_to_dist_2, abs_discount):
    set_of_val= []
    for i,val in phn_to_dist_1.items():
        set_of_val = set_of_val+list(val.keys())
    set_of_val = list(set(set_of_val))
    print(f"codebook enries:{sorted(set_of_val)}")
    print(f"number of codebook entries utilized out of 640:{len(set_of_val)}")
    phn_to_dist_1 = smoothing_dist(phn_to_dist_1, set_of_val, abs_discount)
    phn_to_dist_2 = smoothing_dist(phn_to_dist_2, set_of_val, abs_discount)
    sim_mt = np.zeros((len(phn_to_dist_1), len(phn_to_dist_2)))
    sim_keys_sorted = ['e',
'e??',
 'a??',
 '??',
 '????',
 '????',
 'e??',
 'a??',
 '??',
 '????',
 '??',
 '??',
 '????',
 '????',
 '????',
 'i??',
 'u??',
 '??',
 '????',
 '??',
 'l','r','w',
 'j',
 'm',
 '??',
 'n',
 'b',
 'd',
 'p',
 't',
 'k','??',
 'v','??','??',
 'z',
 'f',
 's', '??',
 '??','t??'
 ]
    sim_keys_sorted = sim_keys_sorted + sorted(list(phn_to_dist_1.keys() - sim_keys_sorted))
    # sim_keys_sorted = phn_to_dist_1.keys()
    for num1, i in enumerate(sim_keys_sorted):
        for num2, j in enumerate(sim_keys_sorted):
            sim_mt[num1][num2] = sim.distance.jensenshannon(
                phn_to_dist_1[i], phn_to_dist_2[j]
            )
            # print(f"sim between {i} and {j}:", sim_mt[num1][num2])
    return sim_mt, sim_keys_sorted

"""calculating similarity with sorted phonemes and creating timit to ipa mappings"""

list(set(phn_dict1.keys())-set(['e',
 'l',
 'j',
 'm',
 'n',
 'b',
 'd',
 'p',
 't',
 'k','??',
 'v',
 'z',
 'f',
 's', '??',
 '??','r']))

abs_discount = 0.00000000002 #absolute discounting hyperparameter
# sim_mt_xlsr,sorted_phonemes = sim.similarity_calculation(phn_dict_final_xlsr, phn_dict_final_xlsr,abs_discount)#passing obtained dictionary with phonemes to codeentries mapping
sim_mt,sorted_phonemes = similarity_calculation(phn_dict1, phn_dict1,abs_discount)
# sim_mt = 1.0-abs(sim_mt_wav2vec2-sim_mt_xlsr)
labels =  sorted_phonemes
print(sorted_phonemes)

"""plotting similarity matrix, dendogram and compression"""

sim.plot_sim(sim_mt, labels, labels,dend = True)
plt.figure()
plot_compress(sim_mt, labels,"TSNE",True)
plt.show()

"""xlsr eng -4, 6, 11, 12, 15, 17, 20, 28, 29, 31, 33, 34, 35, 45, 46, 51, 53, 66, 70, 79, 82, 85, 95, 96, 97, 103, 113, 115, 118, 119, 122, 123, 128, 129, 134, 136, 146, 148, 160, 188, 196, 209, 212, 229, 231, 245, 257, 266, 271, 273, 275, 277, 278, 279, 280, 321, 323, 335, 336, 338, 343, 353, 356, 360, 366, 369, 372, 377, 381, 384, 386, 393, 396, 398, 399, 403, 410, 412, 413, 415, 417, 418, 420, 421, 422, 423, 425, 429, 430, 437, 439, 440, 443, 444, 448, 453, 458, 460, 473, 479, 482, 483, 486, 488, 490, 491, 492, 494, 503, 505, 506, 508, 516, 519, 520, 522, 527, 529, 533, 534, 535, 536, 542, 546, 548, 549, 550, 556, 558, 563, 565, 567, 569, 571, 573, 575, 580, 582, 583, 584, 590, 592, 601, 609, 612, 613, 616, 617, 621, 622, 626, 627, 629, 630, 635(155)

wav2vec2 eng-0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 17, 20, 23, 24, 25, 28, 29, 30, 31, 32, 33, 34, 36, 37, 39, 40, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 68, 69, 72, 73, 77, 78, 80, 82, 83, 85, 88, 89, 91, 92, 95, 96, 100, 101, 102, 105, 108, 109, 111, 112, 113, 114, 115, 118, 120, 121, 123, 124, 125, 128, 129, 130, 131, 132, 133, 136, 137, 138, 143, 145, 146, 148, 150, 151, 154, 156, 158, 159, 162, 163, 165, 168, 170, 171, 173, 174, 177, 181, 182, 183, 184, 185, 186, 187, 189, 190, 191, 197, 201, 202, 204, 205, 206, 209, 210, 212, 215, 216, 217, 218, 219, 221, 223, 224, 225, 229, 231, 232, 234, 237, 240, 242, 245, 246, 247, 249, 250, 253, 254, 258, 259, 261, 262, 263, 264, 266, 270, 271, 272, 273, 274, 275, 277, 278, 279, 280, 282, 283, 285, 287, 288, 289, 290, 292, 294, 298, 299, 300, 302, 303, 304, 306, 307, 308, 310, 311, 313, 316, 317, 318, 319, 320, 322, 325, 326, 327, 328, 329, 333, 335, 336, 338, 340, 341, 344, 346, 349, 350, 352, 353, 354, 355, 356, 360, 362, 366, 367, 371, 372, 373, 376, 378, 379, 383, 384, 390, 392, 395, 397, 398, 399, 401, 402, 403, 404, 407, 410, 411, 417, 418, 420, 422, 423, 424, 426, 427, 430, 431, 432, 435, 436, 443, 445, 448, 449, 450, 452, 453, 455, 457, 460, 461, 462, 463, 465, 467, 468, 469, 472, 474, 475, 477, 478, 479, 483, 484, 486, 487, 488, 490, 491, 492, 494, 495, 497, 499, 501, 504, 506, 507, 508, 513, 514, 517, 518, 519, 520, 522, 523, 524, 527, 528, 529, 531, 532, 534, 535, 536, 537, 538, 540, 541, 546, 548, 551, 553, 554, 556, 558, 560, 561, 565, 568, 570, 571, 572, 573, 574, 575, 576, 578, 580, 581, 583, 584, 586, 588, 589, 592, 593, 594, 599, 603, 604, 605, 606, 607, 609, 610, 611, 613, 614, 621, 622, 625, 627, 630, 631, 633, 636, 637, 638, 639(365)

xlsr ru - 4, 6, 11, 12, 15, 17, 20, 28, 29, 31, 33, 34, 35, 45, 46, 51, 53, 66, 70, 79, 82, 85, 95, 96, 97, 103, 113, 115, 118, 119, 122, 123, 128, 129, 134, 136, 146, 148, 160, 188, 196, 209, 212, 229, 231, 245, 257, 266, 271, 273, 275, 277, 278, 279, 280, 321, 323, 335, 336, 338, 343, 353, 356, 360, 366, 369, 372, 377, 381, 384, 386, 393, 396, 398, 399, 403, 410, 412, 413, 415, 417, 418, 420, 421, 422, 423, 425, 429, 430, 437, 439, 440, 443, 444, 448, 453, 458, 460, 473, 479, 482, 483, 486, 488, 490, 491, 492, 494, 503, 505, 506, 508, 516, 519, 520, 522, 527, 529, 533, 534, 535, 536, 542, 546, 548, 549, 550, 556, 558, 563, 565, 567, 569, 571, 573, 575, 580, 582, 583, 584, 590, 592, 601, 609, 612, 613, 616, 617, 621, 622, 626, 627, 629, 630, 635(155)

wav2vec2 ru - 0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 17, 20, 23, 24, 25, 28, 29, 30, 31, 32, 33, 34, 36, 37, 39, 40, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 68, 69, 72, 73, 77, 78, 80, 82, 83, 85, 88, 89, 91, 92, 95, 96, 100, 101, 102, 105, 108, 109, 111, 112, 113, 114, 115, 118, 120, 121, 123, 124, 125, 128, 129, 130, 131, 132, 133, 136, 137, 138, 143, 145, 146, 148, 150, 151, 154, 156, 158, 159, 162, 163, 165, 168, 170, 171, 173, 174, 177, 181, 182, 183, 184, 185, 186, 187, 189, 190, 191, 197, 201, 202, 204, 205, 206, 209, 210, 212, 215, 216, 217, 218, 219, 221, 223, 224, 225, 229, 231, 232, 234, 237, 240, 242, 245, 246, 247, 249, 250, 253, 254, 258, 259, 261, 262, 263, 264, 266, 270, 271, 272, 273, 274, 275, 277, 278, 279, 280, 282, 283, 285, 287, 288, 289, 290, 292, 294, 298, 299, 300, 302, 303, 304, 306, 307, 308, 310, 311, 313, 316, 317, 318, 319, 320, 322, 325, 326, 327, 328, 329, 333, 335, 336, 338, 340, 341, 344, 346, 349, 350, 352, 353, 354, 355, 356, 360, 362, 366, 367, 371, 372, 373, 376, 378, 379, 383, 384, 390, 392, 395, 397, 398, 399, 401, 402, 403, 404, 407, 410, 411, 417, 418, 420, 422, 423, 424, 426, 427, 430, 431, 432, 435, 436, 443, 445, 448, 449, 450, 452, 453, 455, 457, 460, 461, 462, 463, 465, 467, 468, 469, 472, 474, 475, 477, 478, 479, 483, 484, 486, 487, 488, 490, 491, 492, 494, 495, 497, 499, 501, 504, 506, 507, 508, 513, 514, 517, 518, 519, 520, 522, 523, 524, 527, 528, 529, 531, 532, 534, 535, 536, 537, 538, 540, 541, 546, 548, 551, 553, 554, 556, 558, 560, 561, 565, 568, 570, 571, 572, 573, 574, 575, 576, 578, 580, 581, 583, 584, 586, 588, 589, 592, 593, 594, 599, 603, 604, 605, 606, 607, 609, 610, 611, 613, 614, 621, 622, 625, 627, 630, 631, 633, 636, 637, 638, 639(365)
"""