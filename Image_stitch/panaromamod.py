import cv2
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt


def calibrate_image(img, mtx, dist, newcameramtx, roi):
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst


def crop_black_area(img):
    _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    img = img[y:y + h, x:x + w]
    return img


def stitch_image(panaroma, new_frame, stepnum, countmerge, frame_count, surfparameter, offset, frameresizefactor,
                 par_dict):
    """

    :type panaroma: object
    """
    gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    crop_black_area(gray)
    gray = calibrate_image(gray, **par_dict)
    gray = cv2.resize(gray, (
        int(np.rint(gray.shape[1] / (frameresizefactor))), int(np.rint(gray.shape[0] / (frameresizefactor)))),
                      interpolation=cv2.INTER_AREA)

    surf = cv2.xfeatures2d.SURF_create(surfparameter)
    kp1, des1 = surf.detectAndCompute(gray, None)
    kp2, des2 = surf.detectAndCompute(panaroma, None)

    matches = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]

    print('frame_num: ' + str(frame_count) + ' stepnum: ' + str(stepnum) + ' round: ' + str(
        countmerge) + ' matchs= ' + str(len(good)))

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts + offset, cv2.RANSAC, 1.0)

    resultsize = ((panaroma.shape[1] + 2 * gray.shape[1]), panaroma.shape[0] + 2 * gray.shape[0])

    gray = cv2.warpPerspective(gray, M, resultsize)

    _, newImagethresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    result = np.zeros((gray.shape[0], gray.shape[1]), dtype=np.uint8)
    result[offset:offset + panaroma.shape[0], offset:offset + panaroma.shape[1]] = panaroma

    _, resultthresh = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY)

    overlapppingmask = cv2.bitwise_and(resultthresh, newImagethresh)
    overlapppingmask = cv2.bitwise_not(overlapppingmask)

    gray = cv2.bitwise_and(gray, overlapppingmask)

    panaroma = cv2.add(result, gray)

    panaroma = crop_black_area(panaroma)

    cv2.imwrite('D:\\Archive\\general_codebase\\Image_stitch\\output_image\\componentimage' + str(epoch) + '.png',
                panaroma)

    return len(good), panaroma


if __name__ == '__main__':

    mtx = np.load("D:\\Archive\\general_codebase\\Image_stitch\\calibration_data\\mtx.npz")['arr_0']
    dist = np.load("D:\\Archive\\general_codebase\\Image_stitch\\calibration_data\\dist.npz")['arr_0']
    newcameramtx = np.load("D:\\Archive\\general_codebase\\Image_stitch\\calibration_data\\newcameramtx.npz")['arr_0']
    roi = np.load("D:\\Archive\\general_codebase\\Image_stitch\\calibration_data\\roi.npz")['arr_0']

    parameter_dict = dict(mtx=mtx, dist=dist, newcameramtx=newcameramtx, roi=roi)

    count = 0
    frame_count = 0
    stepnum = 30
    min_keypoint = 70
    countmerge = 0
    countall = 0
    offset = 500
    framesizefactor = 1
    surfparameter = 0
    videodir = 'D:\\Archive\\general_codebase\\Image_stitch\\input_uav_video_data\\sampleuav4.avi'

    cap = cv2.VideoCapture(videodir)
    ret, panaroma = cap.read()

    panaroma = cv2.cvtColor(panaroma, cv2.COLOR_BGR2GRAY)
    panaroma = crop_black_area(panaroma)
    panaroma = calibrate_image(panaroma, **parameter_dict)
    panaroma = cv2.resize(panaroma, (
    int(np.rint(panaroma.shape[1] / (framesizefactor))), int(np.rint(panaroma.shape[0] / (framesizefactor)))),
                          interpolation=cv2.INTER_AREA)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    lastframe = None

    while (cap.isOpened()):
        ret, frame = cap.read()
        frame_count = frame_count + 1
        count = count + 1

        if (not ret):
            countmerge = countmerge + 1
            num, nextpanaroma = stitch_image(panaroma, lastframe, stepnum, countmerge, frame_count, surfparameter,
                                             offset, framesizefactor, parameter_dict)
            panaroma = nextpanaroma
            break

        if frame is not None:
            lastframe = frame
        countall = countall + 1

        if (count % stepnum == 0 and (frame is not None)):
            countmerge = countmerge + 1
            good_match_num, nextpanaroma = stitch_image(panaroma, frame, stepnum, countmerge, frame_count,
                                                        surfparameter, offset, framesizefactor, parameter_dict)
            while (good_match_num < 70):
                surfparameter = 0
                good_match_num, nextpanaroma = stitch_image(panaroma, frame, stepnum, countmerge, frame_count,
                                                            surfparameter, offset, framesizefactor, parameter_dict)
            panaroma = nextpanaroma
            
    panaroma= cv2.medianBlur(panaroma,3)
    cv2.imwrite('D:\\Archive\\general_codebase\\Image_stitch\\output_image\\finalresult.png', panaroma)
