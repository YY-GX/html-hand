


import numpy as np
import pickle
__all__ = ['HTML_numpy']

class HTML_numpy():
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        self.tex_mean = self.model['mean']  # the mean texture
        self.tex_basis = self.model['basis'] # 101 PCA comps
        self.index_map = self.model['index_map'] # the index map, from a compact vector to a 2D texture image

        self.num_total_comp = 101

    def check_alpha(self, alpha):
        # just for checking the alpha's length
        if alpha.size < self.num_total_comp :
            n_alpha = np.zeros(self.num_total_comp,1)
            n_alpha[0:alpha.size,:] = alpha
        elif alpha.size > self.num_total_comp:
            n_alpha = alpha.reshape(alpha.size,1)[0:self.num_total_comp,:]
        else:
            n_alpha = alpha
        return alpha

    def get_mano_texture(self, alpha):
        # first check the length of the input alpha vector
        alpha = self.check_alpha(alpha)
        offsets = np.dot(self.tex_basis, alpha)
        tex_code = offsets + self.tex_mean
        new_tex_img = self.vec2img(tex_code, self.index_map) / 255

        return new_tex_img


    def vec2img(self, tex_code, index_map):
        # inverse vectorize: from compact texture vector to 2D texture image
        img1d = np.zeros(1024*1024*3)
        img1d[index_map] = tex_code
        return img1d.reshape((3, 1024,1024)).transpose(2,1,0)