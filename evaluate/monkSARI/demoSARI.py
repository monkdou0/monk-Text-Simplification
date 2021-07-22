from typing import List
from sari import corpus_sari
def read_lines(filename):
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines



def get_sys_sents(sys_sents_path=None):
    if sys_sents_path is not None:
        return read_lines(sys_sents_path)



def get_orig_and_refs_sents( orig_sents_path=None, refs_sents_paths=None):
    if type(refs_sents_paths) == str:
        refs_sents_paths = refs_sents_paths.split(",")
    orig_sents = read_lines(orig_sents_path)
    refs_sents = [read_lines(ref_sents_path) for ref_sents_path in refs_sents_paths]
    assert all([len(orig_sents) == len(ref_sents) for ref_sents in refs_sents])
    return orig_sents, refs_sents


asset_orig_sents_path = "asset/asset.test.orig"
asset_refs_sents_path = [ f'asset/asset.test.simp.{i}' for i in range(10)]

turk_orig_sents_path = "turkcorpus/legacy/test.8turkers.tok.norm"
turk_refs_sents_path = [f'turkcorpus//legacy/test.8turkers.tok.turk.{i}' for i in range(8)]

def getRes(sys_sents_path):

    sys_sents = get_sys_sents(sys_sents_path)
    # orig_sents, refs_sents = get_orig_and_refs_sents(asset_orig_sents_path, asset_refs_sents_paths)
    orig_sents, refs_sents = get_orig_and_refs_sents(turk_orig_sents_path,turk_refs_sents_path)
    # sys_sents = [x.lower() for x in sys_sents]
    sari_res,add_score,keep_score,del_score = corpus_sari(
        orig_sents=orig_sents, sys_sents=sys_sents, refs_sents=refs_sents
    )
    print(sys_sents_path)
    print("turkcorpus")

    print(sari_res,add_score,keep_score,del_score)

    orig_sents, refs_sents = get_orig_and_refs_sents(asset_orig_sents_path,asset_refs_sents_path)
    # sys_sents = [x.lower() for x in sys_sents]
    sari_res,add_score,keep_score,del_score = corpus_sari(
        orig_sents=orig_sents, sys_sents=sys_sents, refs_sents=refs_sents
    )
    print(sys_sents_path)
    print("turkcorpus")
    print(sari_res,add_score,keep_score,del_score)



    # print(add_score)
    # print(keep_score)
    # print(del_score)


import os
#
# g = os.walk("../output")
#
# for path,dir_list,file_list in g:
#     for file_name in file_list:
#         getRes(os.path.join(path, file_name) )

getRes("pegasus.txt")