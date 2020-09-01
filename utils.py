import cv2

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def rotate_frame(frame, rotation=90):
    (rows, cols) = frame.shape[:2]
    M = cv2.getRotationMatrix2D(center = (cols/2,rows/2), angle=-90, scale=1)
    tran = cv2.warpAffine(frame, M, (cols,rows))
    return tran