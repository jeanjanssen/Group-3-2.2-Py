from ImageReader import ImageReader


def main():
    ir = ImageReader()
    image_path_n = 'C:\\Users\\jeanj\\PycharmProjects\\Group-3-2.2-Py\\Negatives'
    image_path_p = 'C:\\Users\\jeanj\\PycharmProjects\\Group-3-2.2-Py\\Positives'

    image_list_n = ir.load_images_from_folder(image_path_n)
    image_list_p = ir.load_images_from_folder(image_path_p)
    test_p = ir.convert_to_grayscale(image_list_p, image_path_p)
    test_n = ir.convert_to_grayscale(image_list_n)


if __name__ == "__main__":
    main()
