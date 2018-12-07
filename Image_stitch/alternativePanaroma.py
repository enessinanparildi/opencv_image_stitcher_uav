# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 20:48:46 2016

@author: LNU-2942-AP
"""

import cv2
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt


def videoread(videodir):
    cap = cv2.VideoCapture(videodir)
    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        frames.append(gray)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return frames

def calibrateImage (img):
    h,  w = img.shape[:2]
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

def preprocessStream (frames):
    toRet = [calibrateImage(frame) for frame in frames ]
    return toRet

def preprocessSingle (frame):
    return calibrateImage(frame)

def cropBlack (img):
    _,thresh = cv2.threshold(img,1,255,cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    img = img[y:y+h,x:x+w]
    return img

def generatePanaroma(newframes):
    panaroma = newframes[0]
    cv2.imwrite('D:\\Archive\\general_codebase\\Image_stich\\output_image\\baseframe.png', panaroma)
    step = 80
    offset = 1000
    epoch = 0
    for i in range(0,640,step):
            basecutted = panaroma
            epoch = epoch  + 1
            added = newframes[i + step - 1]
            cv2.imwrite('D:\\Archive\\general_codebase\\Image_stitch\\output_image\\componentimage' + str(epoch) + '.png', added)
    
            sift = cv2.xfeatures2d.SIFT_create()
            
            kp1,des1 = sift.detectAndCompute(added,None)
            kp2,des2 = sift.detectAndCompute(basecutted,None)
            
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1,des2, k=2)
            
            
            good= [ m for m,n in matches if m.distance < 0.7*n.distance]

            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            
            M, mask = cv2.findHomography(src_pts , dst_pts+ offset, cv2.RANSAC,1.0)
            resultsize = (( basecutted.shape[1] + 2*added.shape[1] ), basecutted.shape[0] + 2*added.shape[0] )
            dst = cv2.warpPerspective(added,M,resultsize)
            dst[ offset:offset + basecutted.shape[0], offset:offset+basecutted.shape[1]] = basecutted
            panaroma = dst
        
            _,thresh = cv2.threshold(panaroma,1,255,cv2.THRESH_BINARY)
            contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]

            x,y,w,h = cv2.boundingRect(cnt)
            panaroma = panaroma[y:y+h,x:x+w] 
            cv2.imwrite('D:\\Archive\\general_codebase\\Image_stitch\\output_image\\resultiteration' + str(epoch) + '.png', panaroma)
            panaroma = stretch_Img(panaroma)

    #to eliminate some remaining artifacts
    panaroma= cv2.medianBlur(panaroma,3)
    cv2.imwrite('D:\\Archive\\general_codebase\\Image_stitch\\output_image\\finalresult.png', panaroma)
    return panaroma



def stretch_Img(newImage):
   
    _,thresh = cv2.threshold(newImage,1,255,cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    
    maxA = -1
    cnt= None
    for c in contours :
        a = cv2.contourArea(c)
        if a > maxA :
            maxA = a
            cnt = c
    
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    approx.resize(approx.shape[0],approx.shape[2])
    
    
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    
    dst =  [(x,y)]   
    dist = distance.cdist(approx,dst)
    ptleftupper = approx[dist.argmin()]
    
    dst = [(x,y+h)]
    dist = distance.cdist(approx,dst)
    ptrightupper = approx[dist.argmin()]  
                          
    dst = [(w+x,h+y)]
    dist = distance.cdist(approx,dst)
    ptrightdown = approx[dist.argmin()]
                          
    dst = [(w+x,h+y)]
    dist = distance.cdist(approx,dst)
    ptleftdown = approx[dist.argmin()]  
    
    src_pts = np.float32([ ptleftupper,ptrightupper,ptrightdown,ptleftdown]).reshape(-1,1,2)
    dst_pts = np.float32([ [x,y],[x,h+y],[w+x,h+y],[w+x,y] ]).reshape(-1,1,2)
    mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,1.0)
    result = cv2.warpPerspective(newImage,mat,(newImage.shape[1],newImage.shape[0]))
    return result

def generateMSERPanaroma(frames):
    panaroma = frames[0]
    cv2.imwrite('C:/Users/EGT/Desktop/WinPython-64bit-3.5.2.3Qt5b1/videopanaromaimage/baseframe.png', panaroma)    
    step = 60
    offset = 2000
    epoch = 0
    for i in range(0,300,step):
            basecutted = panaroma
            epoch = epoch  + 1
            added = frames[i + step]
            cv2.imwrite('C:/Users/EGT/Desktop/WinPython-64bit-3.5.2.3Qt5b1/videopanaromaimage/componentimage' + str(epoch) + '.png', added)    
            mser = cv2.MSER_create()
            sift = cv2.xfeatures2d.SIFT_create()
            
            kp2 = mser.detect(basecutted, None)
            kp1 = mser.detect(added, None)
            kt1,ds2 = sift.compute(basecutted,kp2)
            kt2,ds1 = sift.compute(added,kp1)
            
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)
            
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(np.asarray(ds1,np.float32),np.asarray(ds2,np.float32),k=2)
            good=[]
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m) 
                        
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            
            M, mask = cv2.findHomography(src_pts , dst_pts+ offset, cv2.RANSAC,1.0)
            resultsize = (( basecutted.shape[1] + 2*added.shape[1] ), basecutted.shape[0] + 2*added.shape[0] )
            dst = cv2.warpPerspective(added,M,resultsize)
            cv2.imwrite('C:/Users/EGT/Desktop/WinPython-64bit-3.5.2.3Qt5b1/videopanaromaimage/warpiteration' + str(epoch) + '.png', dst)         
            
        
            dst[ offset:offset + basecutted.shape[0], offset:offset+basecutted.shape[1]] = basecutted;
            panaroma = dst
        
            _,thresh = cv2.threshold(panaroma,1,255,cv2.THRESH_BINARY)
            contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            x,y,w,h = cv2.boundingRect(cnt)
            panaroma = panaroma[y:y+h,x:x+w] 
            cv2.imwrite('C:/Users/EGT/Desktop/WinPython-64bit-3.5.2.3Qt5b1/videopanaromaimage/resultiteration' + str(epoch) + '.png', panaroma)  
            panaroma = strechImg(panaroma)
    #to eliminate some reamining artifacts
    panaroma= cv2.medianBlur(panaroma,3)
    cv2.imwrite('C:/Users/EGT/Desktop/WinPython-64bit-3.5.2.3Qt5b1/videopanaromaimage/result.png', panaroma) 
    return panaroma


def adaptivePanaroma(frames,maxFrameNum):
    panaroma = frames[0]
    cv2.imwrite('C:/Users/EGT/Desktop/WinPython-64bit-3.5.2.3Qt5b1/videopanaromaimage/baseframe.png', panaroma)    
    basestep = 50
    step = 50
    offset = 2000
    epoch = 0
    MIN_MATCH_COUNT = 20
    frameNum = step
    maxFrame = maxFrameNum
    while(frameNum != maxFrame):
        epoch = epoch  + 1
        added = frames[frameNum - 1]
        basecutted = panaroma
        cv2.imwrite('C:/Users/EGT/Desktop/WinPython-64bit-3.5.2.3Qt5b1/videopanaromaimage/componentimage' + str(epoch) + '.png', added)    
        sift = cv2.xfeatures2d.SIFT_create()
            
        kp1,des1 = sift.detectAndCompute(added,None)
        kp2,des2 = sift.detectAndCompute(basecutted,None)
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)
        
        good=[]
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m) 
        
        print('matches :' + str(len(good)))
        if len(good) > MIN_MATCH_COUNT:
            step = basestep            
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            
            M, mask = cv2.findHomography(src_pts , dst_pts+ offset, cv2.RANSAC,1.0)
            resultsize = (( basecutted.shape[1] + 2*added.shape[1] ), basecutted.shape[0] + 2*added.shape[0] )
            dst = cv2.warpPerspective(added,M,resultsize)
            cv2.imwrite('C:/Users/EGT/Desktop/WinPython-64bit-3.5.2.3Qt5b1/videopanaromaimage/warpiteration' + str(epoch) + '.png', dst)         
            
        
            dst[ offset:offset + basecutted.shape[0], offset:offset+basecutted.shape[1]] = basecutted;
            panaroma = dst
        
            _,thresh = cv2.threshold(panaroma,1,255,cv2.THRESH_BINARY)
            contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            x,y,w,h = cv2.boundingRect(cnt)
            panaroma = panaroma[y:y+h,x:x+w] 
            cv2.imwrite('C:/Users/EGT/Desktop/WinPython-64bit-3.5.2.3Qt5b1/videopanaromaimage/resultiteration' + str(epoch) + '.png', panaroma)  
            panaroma = strechImg(panaroma)
            if(maxFrame - step > frameNum ):
                frameNum = frameNum + step
            else:
                frameNum = maxFrame
        else:
            print('Failed Match')
            step = step//2
            frameNum = frameNum - step 
        print(frameNum)
    cv2.imwrite('C:/Users/EGT/Desktop/WinPython-64bit-3.5.2.3Qt5b1/videopanaromaimage/result.png', panaroma)
    return panaroma

def main():
    videodir = 'D:\\Archive\\general_codebase\\Image_stitch\\input_uav_video_data.mp4'
    all_frames =  videoread(videodir)

        
        
        
        
        
        