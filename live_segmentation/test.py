from utils import *
import glob


def get_mask(im):
    """

    :param im: BGR image
    :return :binary mask of dimension 1280,720
    """
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(im)
    im = im.convert('RGB')
    image = transform(im)
    image_tensor = torch.unsqueeze(image, 0).cuda()

    # Get binary mask from the image
    seg_mask = seg_model(image_tensor)
    seg_mask = seg_mask.cpu().detach().numpy().squeeze(0)
    seg_mask[seg_mask > 0] = 1
    seg_mask[seg_mask < 0] = 0

    return cv2.resize(seg_mask, (1280, 720))


def get_dice(image_a, image_b):
    """
    Calculate the Dice coefficient on the given Image
    :param image_a: coefficient image1
    :param image_b: coefficient image2
    :return: A number which tells the dice coefficient on given images.
    """
    return (np.sum(image_b*image_a)*2)/np.sum(image_a+image_b)


def read_image(path, gray=False):
    """

    :param path: path of image to be loaded
    :param gray: True if output required is grayscale
    :return:
    """
    if gray:
        image = cv2.imread(path, 0)
        image[image < 30] = 0
        image[image >= 30] = 1
        return image

    return cv2.imread(path)


def test_case():
    """
    :assert that dice score of sample images is > 0.80
    :return:
    """
    images = glob.glob("test_data/images/*")
    for image in images:
        rgb = read_image(image)
        label = read_image(image.replace("images", "labels"), True)
        mask = get_mask(rgb)

        try:
            assert get_dice(mask, label) > 0.80
            print ("Dice Score test passed, image", image)
        except:
            print ("Dice Score test failed, score less than 0.80", image)


if __name__ == '__main__':
    images = "test_data/images/*"
    labels = "test_set/labels"
    weights_path = "weights/UNET_2.pt"
    transform = get_transforms(512)
    print('Loaded transforms')
    seg_model = get_segmentation_model(weights_path)
    seg_model.eval()

    test_case()
