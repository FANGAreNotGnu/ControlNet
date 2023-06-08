import random


class VisPriorDistribution():
    pass


# NoClip: bbox not going out of image bound, thus no clip of bbox needed
class UniformRandomNoClipVPD(VisPriorDistribution):
    def __init__(self, scale_down = 0.5, scale_up = 2):
        self.scale_down = scale_down
        self.scale_up = scale_up

    def generate_bbox(self, img, bbox):

        x, y, w, h = bbox
        H, W, C = img.shape

        assert W >= 0
        assert H >= 0
        assert w >= 0
        assert h >= 0

        w_max = min(w * self.scale_up, W-1)
        h_max = min(h * self.scale_up, H-1)
        w_min = w * self.scale_down
        h_min = h * self.scale_down

        w_new = int(random.uniform(w_min, w_max))
        h_new = int(random.uniform(h_min, h_max))

        x_max = W - w_new - 1
        y_max = H - h_new - 1    

        x_new = int(random.random() * x_max)
        y_new = int(random.random() * y_max)

        return x_new, y_new, w_new, h_new,

