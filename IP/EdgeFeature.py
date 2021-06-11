class EdgeFeature:
    def __init__(self, h, w, type):
        self.h = h
        self.w = w
        self.type = type

    # Setters
    def set_h(self, h):
        self.h = h

    def set_w(self, w):
        self.w = w

    def set_type(self, type):
        self.type = type

    # 0 0 1 1  1 1 0 0  0 0 0 0  1 1 1 1
    # 0 0 1 1  1 1 0 0  0 0 0 0  1 1 1 1
    # 0 0 1 1  1 1 0 0  1 1 1 1  0 0 0 0
    # 0 0 1 1, 1 1 0 0, 1 1 1 1, 0 0 0 0
    # type 1 , type 2 , type 3 , type 4
    # TODO : need to check if feature is smaller than image, can't have the same size or be bigger

    # Gets the value of the intensity change between pixels
    # x : x position
    # y : y position
    # integral_im : integral image of the image we apply the feature to
    # type defined before
    def get_score(self, x, y, integral_im):

        if self.type == 1:
            p_l = int(round((self.w) / 2)) * self.h
            p_d = (self.w * self.h) - p_l
            sum_light = integral_im[x + int(round((self.w) / 2)), y + self.h] - integral_im[x - 1, y + self.h] - \
                        integral_im[x + int(round((self.w) / 2)), y - 1] + integral_im[x - 1, y - 1]
            sum_dark = integral_im[x + self.w, y + self.h] - integral_im[x + int(round((self.w) / 2)), y + self.h] - \
                       integral_im[x + self.w, y - 1] + integral_im[x + int(round((self.w) / 2)) - 1, y - 1]
            return (sum_dark / p_d) - (sum_light / p_l)

        elif self.type == 2:
            p_d = int(round((self.h) / 2)) * self.w
            p_l = (self.w * self.h) - p_d
            sum_dark = integral_im[x + int(round((self.w) / 2)), y + self.h] - integral_im[x - 1, y + self.h] - \
                       integral_im[x + int(round((self.w) / 2)), y - 1] + integral_im[x - 1, y - 1]
            sum_light = integral_im[x + self.w, y + self.h] - integral_im[x + int(round((self.w) / 2)), y + self.h] - \
                        integral_im[x + self.w, y - 1] + integral_im[x + int(round((self.w) / 2)) - 1, y - 1]
            return (sum_dark / p_d) - (sum_light / p_l)

        elif self.type == 3:
            p_l = int(round((self.h) / 2)) * self.w
            p_d = (self.w * self.h) - p_l
            sum_light = integral_im[x + self.w, y + int(round((self.h) / 2))] - integral_im[
                x - 1, y + int(round((self.h) / 2))] - integral_im[x + self.w, y - 1] + integral_im[x - 1, y - 1]
            sum_dark = integral_im[x + self.w, y + self.h] - integral_im[x - 1, y + self.h] - integral_im[
                x + self.w, y + int(round((self.h) / 2))] + integral_im[x - 1, y + int(round((self.h) / 2))]
            return (sum_dark / p_d) - (sum_light / p_l)

        else:
            p_d = int(round((self.h) / 2)) * self.w
            p_l = (self.w * self.h) - p_d
            sum_light = integral_im[x + self.w, y + int(round((self.h) / 2))] - integral_im[
                x - 1, y + int(round((self.h) / 2))] - integral_im[x + self.w, y - 1] + integral_im[x - 1, y - 1]
            sum_dark = integral_im[x + self.w, y + self.h] - integral_im[x - 1, y + self.h] - integral_im[
                x + self.w, y + int(round((self.h) / 2))] + integral_im[x - 1, y + int(round((self.h) / 2))]
            return (sum_dark / p_d) - (sum_light / p_l)
