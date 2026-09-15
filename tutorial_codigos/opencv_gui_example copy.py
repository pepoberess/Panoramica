import cv2
from matplotlib import pyplot as plt

def example_01():
    """

        Carga una imagen y la muestra en la GUI

    """

    image = cv2.imread('res/frog.jpg')

    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()

    cv2.namedWindow("image", cv2.WINDOW_FREERATIO)
    cv2.imshow("image", image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()



def example_02():
    """

        Abre un dispositivo de captura de video y muestra los frames en la GUI

    """

    source = 1
    cap = cv2.VideoCapture(source)
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = 255 - frame

        cv2.imshow("frame", frame)

        key = cv2.waitKey(1)  # & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # example_01()
    example_02()
