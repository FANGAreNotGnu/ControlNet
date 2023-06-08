import random

class VisPriorDistribution():
    pass

class UniformRandomLocationNoClipVPD(VisPriorDistribution):
    def generate_bbox(self, W, H, x, y, w, h):
        # W, H: image
        # x, y, w, h: bbox
        # NoClip: bbox not going out of image bound, thus no clip of bbox needed
        assert W >= 0
        assert H >= 0
        assert w >= 0
        assert h >= 0

        x_max = W - w - 1
        y_max = H - h - 1    

        x_new = random.random() * x_max
        y_new = random.random() * y_max

        return x_new,y_new,w,h

