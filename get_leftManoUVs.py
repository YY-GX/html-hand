# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 20:04:07 2020

@author: qiann
"""

# goal: generate a new faces_uv for the left MANO model
# procedure: 1. build the face index map, since the face index is different for
# the left hand and right hand. For example, the first face in left is different from
# right hand.
 
import numpy as np
import pickle
import collections
#collections.Counter(x) == collections.Counter(y)
# Load the MANO_RIGHT.pkl
right_path = './MANO_RIGHT.pkl'
with open(right_path, 'rb') as f:
    MANO = pickle.load(f, encoding='latin')
rightFaces = MANO['f']

left_path = './MANO_LEFT.pkl'
with open(left_path, 'rb') as f:
    MANO = pickle.load(f, encoding='latin')
leftFaces = MANO['f']
leftVerts = MANO['v_template']

uv_path = './TextureBasis/uvs_right.pkl'
with open(uv_path, 'rb') as f:
    manoUVs = pickle.load(f, encoding = 'latin')
rfaces_uv = manoUVs['faces_uvs'] 
verts_uv = manoUVs['verts_uvs']


fmap = list()
for lf in leftFaces:
    # for each face in leftFaces, find its corresponding face in right hand
    for i , rf in enumerate(rightFaces):
        if(collections.Counter(rf) == collections.Counter(lf)):
            fmap.append(i)
            break

#print(fmap)

# next we thus can generate the new faces_uv with the fmap, but still require the vertex Id
lfaces_uv = np.copy(rfaces_uv)
for i in range(len(lfaces_uv)):
    # first obtain the corresponding face index in right
    ri = fmap[i]

    rFace = rightFaces[ri]
    lFace = leftFaces[i]
    rFaceUV = rfaces_uv[ri]
    uvList = list()
    for lvi in lFace:
        pos = np.where(rFace == lvi)[0]
        uvList.append(rFaceUV[pos])
    lFaceUV = np.array(uvList).reshape(3)    
    lfaces_uv[i] = lFaceUV
# #print(lfaces_uv[1244])

leftUVs = dict()
leftUVs['faces_uvs'] = lfaces_uv
leftUVs['verts_uvs'] = verts_uv


def storeUVs2Pickle(uvs, path):
     pickle.dump(uvs, open(path, 'wb'), -1)


#def storeObj( outmesh_path, verts, faces, verts_uv = None, faces_uv = None):
#    with open(outmesh_path, 'w') as fp:
#        fp.write('mtllib vis.mtl\n')
#        for v in verts:
#            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
#        if not (verts_uv is None):
#            for t in verts_uv:
#                fp.write('vt %f %f\n' % (t[0], t[1]))
#        
#        if faces_uv is None:
#            for f in faces+1: # Faces are 1-based, not 0-based in obj files
#                fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
#        else:
#            for f, ft in zip(faces + 1, faces_uv+1):
#                fp.write('f %d/%d %d/%d %d/%d\n' % (f[0],ft[0],f[1],ft[1],f[2],ft[2]))

# storeObj('./visLeft.obj', leftVerts, leftFaces, verts_uv, lfaces_uv)  
storeUVs2Pickle(leftUVs, './leftUVs.pkl')      

# test for the new left uv pickle file
#left_uv_path = './leftUVs.pkl'   
#with open(left_uv_path, 'rb') as f:
#    manoUVs = pickle.load(f, encoding = 'latin')
#lfaces_uv = manoUVs['faces_uvs'] 
#verts_uv = manoUVs['verts_uvs']
#storeObj('./visLeft.obj', leftVerts, leftFaces, verts_uv, lfaces_uv)  

