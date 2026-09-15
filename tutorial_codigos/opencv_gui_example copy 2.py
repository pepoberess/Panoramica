import numpy as np
import cv2


def detect_edges_gradient(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 1)

    gx = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    gy = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)

    # normalize magnitude between 0 and 255
    mag = (255 * mag / mag.max()).astype("uint8")

    edges = np.zeros_like(mag)
    edges[mag > 16] = 255
    #edges = mag
    return edges


def detect_edges_canny(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gray = cv2.GaussianBlur(gray, (5, 5), 1)

    th1 = cv2.getTrackbarPos("th1", "edges")
    th2 = cv2.getTrackbarPos("th2", "edges")
    # th1 = 64
    # th2 = 100
    gray = cv2.blur(gray, (5, 5))
    ret = cv2.Canny(gray, th1, th2)

    return ret


def example_edges(edge_detector):
    source = 1
    cap = cv2.VideoCapture(source)
    cv2.namedWindow("edges", cv2.WINDOW_NORMAL)

    nothing = lambda x: x
    cv2.createTrackbar('th1', 'edges', 127, 255, nothing)
    cv2.createTrackbar('th2', 'edges', 0, 255, nothing)

    while True:
        ret, frame = cap.read()

        edges = edge_detector(frame)
        cv2.imshow("edges", edges)

        key = cv2.waitKey(1)  # & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # edge_detector = detect_edges_gradient
    edge_detector = detect_edges_canny

    example_edges(edge_detector)
