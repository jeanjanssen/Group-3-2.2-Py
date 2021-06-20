class LineFeature:

    def __init__(self,h ,w, type):
        self.h = h
        self.w = w
        self.type = type

    #Setters
    def set_h(self, h):
        self.h = h

    def set_w(self, w):
        self.w = w

    def set_type(self, type):
        self.type = type

    # 0 1 0  1 0 1  0 0 0  1 1 1
    # 0 1 0  1 0 1  1 1 1  0 0 0
    # 0 1 0, 1 0 1, 0 0 0, 1 1 1
    # type1, type2, type3, type4
    # TODO : need to check if feature is smaller than image, can't have the same size or be bigger

    # Gets the value of the intensity change between pixels
    # x : x position
    # y : y position
    # integral_im : integral image of the image we apply the feature to
    # type defined before

    def get_score(self, x, y, integral_im):
        if x <= len(integral_im[0]) and y <= len(integral_im):
            if self.type == 1:
                l_p = (int(round((self.w)/3)) * self.h)*2
                d_p = (self.w * self.h) - l_p
                sum_light = (integral_im[y + self.h, x + int(round((self.w) / 3))] - integral_im[y + self.h, x - 1] - integral_im[y - 1, x + int(round((self.w) / 3))] + integral_im[y - 1, x - 1]) + (integral_im[y + self.h, x + self.w] - integral_im[y + self.h, x + int(round(((self.w) * 2) / 3))] - integral_im[y - 1, x + self.w] + integral_im[y - 1, x + int(round(((self.w) * 2) / 3))])
                sum_dark = integral_im[y + self.h, x + int(round((self.w) / 3))] - integral_im[y + self.h, x + int(round((self.w) / 3))] - integral_im[y - 1, x + int(round(((self.w) * 2) / 3))] + integral_im[y - 1, x + int(round((self.w) / 3))]
                return ((sum_dark / d_p) - (sum_light / l_p))

            elif self.type == 2:
                d_p = (int(round((self.w) / 3)) * self.h) * 2
                l_p = (self.w * self.h) - d_p
                sum_dark = (integral_im[y + self.h, x + int(round((self.w) / 3))] - integral_im[y + self.h, x - 1] - integral_im[y - 1, x + int(round((self.w) / 3))] + integral_im[y - 1, x - 1]) + (integral_im[y + self.h, x + self.w] - integral_im[y + self.h, x + int(round(((self.w) * 2) / 3))] - integral_im[y - 1, x + self.w] +integral_im[y - 1, x + int(round(((self.w) * 2) / 3))])
                sum_light = integral_im[y + self.h, x + int(round((self.w) / 3))] - integral_im[y + self.h, x + int(round((self.w) / 3))] - integral_im[y - 1, x + int(round(((self.w) * 2) / 3))] + integral_im[y - 1, x + int(round((self.w) / 3))]
                return ((sum_dark / d_p) - (sum_light / l_p))

            elif self.type == 3:
                l_p = (int(round((self.h) / 3)) * self.w) * 2
                d_p = (self.w * self.h) - l_p
                sum_light = (integral_im[y + int(round((self.h) / 3)), x + self.w] - integral_im[y + int(round((self.h) / 3)), x - 1] - integral_im[y - 1, x + self.w] + integral_im[y - 1, x - 1]) + (integral_im[y + self.h, x + self.w] - integral_im[y + self.h, x - 1] - integral_im[y + int(round(((self.h) * 2) / 3)), x + self.w] + integral_im[y + int(round(((self.h) * 2) / 3)), x - 1])
                sum_dark = integral_im[y + int(round(((self.h) * 2) / 3)), x + self.w] - integral_im[y + int(round(((self.h) * 2) / 3)), x - 1] - integral_im[y + int(round((self.h) / 3)), x + self.w] + integral_im[y + int(round((self.h) / 3)), x - 1]
                return ((sum_dark / d_p) - (sum_light / l_p))

            else:
                d_p = (int(round((self.h) / 3)) * self.w) * 2
                l_p = (self.w * self.h) - d_p
                sum_dark = (integral_im[y + int(round((self.h) / 3)), x + self.w] - integral_im[y + int(round((self.h) / 3)), x - 1] - integral_im[y - 1, x + self.w] + integral_im[y - 1, x - 1]) + (integral_im[y + self.h, x + self.w] - integral_im[y + self.h, x - 1] - integral_im[y + int(round(((self.h) * 2) / 3)), x + self.w] + integral_im[y + int(round(((self.h) * 2) / 3)), x - 1])
                sum_light = integral_im[y + int(round(((self.h) * 2) / 3)), x + self.w] - integral_im[y + int(round(((self.h) * 2) / 3)), x - 1] - integral_im[y + int(round((self.h) / 3)), x + self.w] + integral_im[y + int(round((self.h) / 3)), x - 1]
                return ((sum_dark / d_p) - (sum_light / l_p))

        else:
            return print("Wrong size of feature")
