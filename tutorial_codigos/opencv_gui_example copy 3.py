import numpy as np
import cv2


class ORB:

    def __init__(
            self,
            object_image
    ):

        self.algorithm = cv2.ORB_create(
            nfeatures=2048,
        )
        self.prepare(object_image)

    def prepare(self, object_image):

        pp_object_image = self.preprocess(object_image)
        kps, descriptors = self.detect_and_compute(pp_object_image)

        self.object_image = object_image
        self.obj_kps = kps
        self.obj_descriptors = descriptors

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)

    def detect_and_compute(self, image):

        # For each scene image, compute keypoints and descriptors
        keypoints, descriptors = self.algorithm.detectAndCompute(image, None)
        return keypoints, descriptors

    def detect(self, scene_image):

        pp_scene_image = self.preprocess(scene_image)

        obj_kps = self.obj_kps

        obj_descriptors = self.obj_descriptors
        obj_image = self.object_image

        scene_kps, scene_descriptors = self.detect_and_compute(pp_scene_image)

        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,  # 12
            key_size=12,  # 20
            multi_probe_level=1)  # 2

        search_params = dict(checks=50)  # or pass empty dictionary

        # Create FLANN based matcher object
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        k = 2
        matches = flann.knnMatch(obj_descriptors, scene_descriptors, k)
        # Need to draw only good matches, so create a mask
        matches_mask = [[0, 0] for i in range(len(matches))]

        #
        # ratio test
        for i in range(len(matches)):
            match = matches[i]
            if len(match) != 2:
                continue
            m, n = matches[i]

            if m.distance < 0.6 * n.distance:
                matches_mask[i] = [1, 0]

        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matches_mask, flags=0)
        res_img = cv2.drawMatchesKnn(
            obj_image,
            obj_kps,
            scene_image,
            scene_kps,
            matches,
            None,
            **draw_params
        )

        return res_img


class SIFT:

    def __init__(
            self,
            object_image
    ):

        self.algorithm = cv2.SIFT_create(
            nfeatures=1000,
        )

        self.prepare(object_image)

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)

    def prepare(self, object_image):

        pp_object_image = self.preprocess(object_image)

        kps, descriptors = self.detect_and_compute(
            pp_object_image
        )

        self.object_image = object_image
        self.obj_kps = kps
        self.obj_descriptors = descriptors

        if len(kps) > 0:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            flann.add([descriptors])
            flann.train()
            self.flann = flann
        else:
            self.flann = None

    def detect_and_compute(self, image):

        # For each scene image, compute keypoints and descriptors
        keypoints, descriptors = self.algorithm.detectAndCompute(image, None)
        return keypoints, descriptors

    def detect(self, scene_image):

        pp_scene_image = self.preprocess(scene_image)

        obj_kps = self.obj_kps
        obj_descriptors = self.obj_descriptors
        obj_image = self.object_image

        scene_kps, scene_descriptors = self.detect_and_compute(pp_scene_image)

        flann = self.flann

        if flann:
            matches = flann.knnMatch(obj_descriptors, scene_descriptors, k=2)

        else:
            matches = []

        matches_mask = [[0, 0] for i in range(len(matches))]
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.6 * n.distance:
                matches_mask[i] = [1, 0]

        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matches_mask, flags=0)
        res_img = cv2.drawMatchesKnn(
            obj_image,
            obj_kps,
            scene_image,
            scene_kps,
            matches,
            None,
            **draw_params
        )

        return res_img




def live_feats(method):

    source = 1
    cap = cv2.VideoCapture(source)

    #cv2.namedWindow("object", cv2.WINDOW_NORMAL)
    #cv2.imshow("object", method.object_image)

    while True:
        ret, frame = cap.read()

        #cv2.imshow("frame", frame)

        detection = method.detect(frame)
        cv2.imshow("detection", detection)

        key = cv2.waitKey(5)  # & 0xFF
        if key == ord("q"):
            break

        elif key == ord("o"):

            method.prepare(frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # img_obj = cv2.imread("res/yaguarete.png")
    # img_obj = cv2.imread("res/guanaco.jpg")
    # img_obj = cv2.imread("res/20dolars.jpg")
    img_obj = np.zeros((720, 1280, 3), np.uint8)

    # nz = 640
    # w, h = img_obj.shape[1], img_obj.shape[0]
    # nh = int(nz * h / w)
    # img_obj = cv2.resize(img_obj, (nz, nh))

    method = ORB(img_obj)
    # method = SIFT(img_obj)

    live_feats(method)
