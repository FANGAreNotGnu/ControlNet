import random


class VisPriorLayout():
    pass


# NoClip: bbox not going out of image bound, thus no clip of bbox needed
class UniformRandomNoClipVPL(VisPriorLayout):
    def __init__(self, scale_down = 1, scale_up = 1):
        self.scale_down = scale_down
        self.scale_up = scale_up

    def generate_a_layout_with_prior(self, im_shape, priors, num_object):
        layout = []
        object_categories = random.sample(priors.keys(), num_object)
        for category_name in object_categories:
            im_prior = random.choice(priors[category_name])
            bbox = self.generate_bbox(im_shape=im_shape, prior_shape=im_prior.shape[:2])
            layout.append([category_name, bbox, im_prior])
        
        return layout

    def generate_layouts_with_prior(self, im_shape, priors, num_object, num_layouts):
        layouts = []
        for i in range(num_layouts):
            layout = self.generate_a_layout_with_prior(im_shape, priors, num_object)
            layouts.append(layout)

        return layouts

    def generate_bbox(self, im_shape, prior_shape):

        h, w = prior_shape
        H, W, _ = im_shape

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

