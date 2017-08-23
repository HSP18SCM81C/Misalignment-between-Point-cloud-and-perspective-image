import numpy as np
import math
import cv2
import misalignment_angle
from skimage.filters import threshold_adaptive

csv = open('final_project_point_cloud.fuse', 'rb')
point_info = []
point_cloud_display = []
#frontimage = [[0]*2048]*2048
frontimage = np.zeros((2048,2048), dtype = float)
frontimage2 = np.zeros((2048,2048), dtype = float)
backImage = np.zeros((2048,2048), dtype = float)
rightImage = np.zeros((2048,2048), dtype = float)
leftImage = np.zeros((2048,2048), dtype = float)
#converts lat, lon, elevation into xyz coordinates
#source: http://stackoverflow.com/questions/28365948/javascript-latitude-longitude-to-xyz-on-sphere-three-js
def cartesian(lat,lon, elevation):
    cosLat = (math.cos(lat * math.pi / 180.0))
    sinLat = (math.sin(lat * math.pi / 180.0))
    cosLon = (math.cos(lon * math.pi / 180.0))
    sinLon = (math.sin(lon * math.pi / 180.0))
    a=6378137.0 + elevation
    f=1.0 / 298.257224
    e=math.sqrt(f*(2-f))
    N=a/math.sqrt(1-e*e*sinLat*sinLat)
    x=(N + elevation) * cosLon * cosLat
    y=(N + elevation) * cosLon * sinLat
    z=((1-e*e)*N + elevation) * sinLon
    return x,y,z,cosLat, cosLon, sinLat, sinLon
 

def ecef2enu(lat,lon, elevation, cosLat, cosLon, sinLat, sinLon):
    x1, y1, z1, cosLat1, cosLon1, sinLat1, sinLon1 = cartesian(45.90414414, 11.02845385,227.5819)
    dx = lat - x1;
    dy = lon - y1;
    dz = elevation - z1;
    cosLat = cosLat
    sinLat = sinLat
    cosLon = cosLon
    sinLon = sinLon
    r = [[-sinLon, cosLon, 0], [-(cosLon * sinLat), -(sinLat * sinLon), cosLat], [cosLon*cosLat, cosLat*sinLon, sinLat]]
    po = [dx,dy,dz]
    penu = np.dot(r, po)
    #print(penu)
    x = penu[0]
    y = penu[1]
    z = penu[2]
    return x, y, z


def enu2cc(e,n, u):

    qs = 0.362114
    qx = 0.374050
    qy = 0.592222
    qz = 0.615007
    a =  (qs*qs) + (qx*qx) - (qy*qy) - (qz*qz)
    b =  (2*qx*qy) - (2*qs*qz)
    c =  (2*qx*qz) + (2*qs*qy)
    d =  (2*qx*qy) + (2*qs*qz)
    e1 =  (qs*qs) - (qx*qx) + (qy*qy) - (qz*qz)
    f =  (2*qz*qy) - (2*qs*qx)
    g =  (2*qx*qz) - (2*qs*qy)
    h =  (2*qz*qy) + (2*qs*qx)
    i =  (qs*qs) - (qx*qx) - (qy*qy) + (qz*qz)
    rq = [[a, b, c], [d,e1,f], [g,h,i]]
    pneu = [n,e,-u]
    cc = np.dot(rq, pneu)
    #print(penu)
    x = cc[0]
    y = cc[1]
    z = cc[2]
    return x, y, z


#file in form of: [latitude] [longitude] [altitude] [intensity]
#write original point cloud data into a file
for line in csv:
    r = line.decode('utf8').strip().split(' ')
    point = []
    x, y, z, cosLat, cosLon, sinLat, sinLon = cartesian(float(r[0]), float(r[1]), float(r[2]))
    point.append(x)
    point.append(y)
    point.append(z)
    point.append(cosLat)
    point.append(cosLon)
    point.append(sinLat)
    point.append(sinLon)
    point_info.append(point)

initial = open('pointcloud.obj', 'w')
penufile = open('penufile.obj', 'w')
ccfile = open('ccfile.obj', 'w')
fffile = open('frontface.obj', 'w')
bffile = open('backface.obj', 'w')
rffile = open('rightface.obj', 'w')
lffile = open('leftface.obj', 'w')
#write file in necessary format for .obj
x1, y1, z1, cosLat1, cosLon1, sinLat1, sinLon1 = cartesian(45.90414414, 11.02845385,227.5819)
xe, yn, zu= ecef2enu(float(x1), float(y1), float(z1), float(cosLat1), float(cosLon1), float(sinLat1),float(sinLon1))
xc, yc, zc = enu2cc(xe,yn,zu)
for point in point_info:
    line = "v " + str(point[0]) + " " + str(point[1]) + " "+ str(point[2])
    initial.write(line)
    initial.write("\n")
    e,n,u = ecef2enu(float(point[0]), float(point[1]), float(point[2]), float(point[3]), float(point[4]), float(point[5]),float(point[6]))
    line2 = "v " + str(e) + " " + str(n) + " "+ str(u)
    penufile.write(line2)
    penufile.write("\n")
    cc1, cc2, cc3 = enu2cc(e,n,u)
    #cc1, cc2, cc3 = cc1-math.radians(30), cc2-math.radians(30), cc3-math.radians(30)
    line3 = "v " + str(cc1) + " " + str(cc2) + " " + str(cc3)
    if (cc3 > 0) & (cc3 > abs(cc1)) & (cc3 > abs(cc2)):
        line4 = "v " + str(cc1) + " " + str(cc2) + " " + str(cc3)
        fffile.write(line4)
        fffile.write("\n")
        xi = int(((cc2/cc3)*((2048-1)/2)) + ((2048+1)/2))
        yi=  int(((cc1/cc3)*((2048-1)/2)) + ((2048+1)/2))
        F = cc3-zc;
        #xi2 = int((((cc1 - xc) * (F/cc3)) + xc ) * ((2047/2)+(2049/2)) )
        #yi2 = int((((cc2 - yc) * (F/cc3)) + yc ) * ((2047/2)+(2049/2)) )
        frontimage[xi][yi]=255;
    if (cc1 > 0) & (cc1 > abs(cc3)) & (cc1 > abs(cc2)):
        line5 = "v " + str(cc1) + " " + str(cc2) + " " + str(cc3)
        rffile.write(line5)
        rffile.write("\n")
        xi = int(((cc2/cc1)*((2048-1)/2)) + ((2048+1)/2))
        yi=  int(((cc3/cc1)*((2048-1)/2)) + ((2048+1)/2))
        rightImage[xi][yi]=255;
    if (cc3 < 0) & (abs(cc3) > abs(cc1)) & (abs(cc3) > abs(cc2)):
        line6 = "v " + str(cc1) + " " + str(cc2) + " " + str(cc3)
        bffile.write(line6)
        bffile.write("\n")
        xi = int(-((cc2/cc3)*((2048-1)/2)) + ((2048+1)/2))
        yi=  int(-((cc1/cc3)*((2048-1)/2)) + ((2048+1)/2))
        backImage[xi][yi]=255;
    if (cc1 < 0) & (abs(cc1) > abs(cc3)) & (abs(cc1) > abs(cc2)):
        line7 = "v " + str(cc1) + " " + str(cc2) + " " + str(cc3)
        lffile.write(line7)
        lffile.write("\n")
        xi = int(-((cc2/cc1)*((2048-1)/2)) + ((2048+1)/2))
        yi=  int(-((cc3/cc1)*((2048-1)/2)) + ((2048+1)/2))
        leftImage[xi][yi]=255;
    ccfile.write(line3)
    ccfile.write("\n")

cv2.imwrite('front.png',frontimage)
cv2.imwrite('front2.png',frontimage2)
cv2.imwrite('back.png',backImage)
cv2.imwrite('right.png',rightImage)
cv2.imwrite('left.png',leftImage)

img2 = cv2.imread("front.jpg");
img1 = cv2.imread("front.png");
img4 = cv2.imread("right.jpg");
img3 = cv2.imread("right.png");
img6 = cv2.imread("back.jpg");
img5 = cv2.imread("back.png");
img8=cv2.imread("left.jpg")
img7=cv2.imread("left.png")

orb = cv2.ORB_create(1000, 1.2)

# Detect keypoints of original image
(kp1, des1) = orb.detectAndCompute(img1, None)
(kp3, des3) = orb.detectAndCompute(img3, None)
(kp5, des5) = orb.detectAndCompute(img5, None)
(kp7, des7) = orb.detectAndCompute(img7, None)

# Detect keypoints of rotated image
(kp2, des2) = orb.detectAndCompute(img2, None)
(kp4, des4) = orb.detectAndCompute(img4, None)
(kp6, des6) = orb.detectAndCompute(img6, None)
(kp8, des8) = orb.detectAndCompute(img8, None)

# Create matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Do matching
matches1 = bf.match(des1, des2)
matches2 = bf.match(des3, des4)
matches3 = bf.match(des5, des6)
matches4 = bf.match(des7, des8)


matches1 = sorted(matches1, key=lambda val: val.distance)
matches2 = sorted(matches2, key=lambda val: val.distance)
matches3 = sorted(matches3, key=lambda val: val.distance)
matches4 = sorted(matches4, key=lambda val: val.distance)

ccfile.close()
# Show only the top 10 matchesfffile.close()
print ("The misalignment of front image are:")
misalignment_angle.drawMatches(img1, kp1, img2, kp2, matches1)
print ("The misalignment of right image are:")

misalignment_angle.drawMatches(img3, kp3, img4, kp4, matches2)
print ("The misalignment of back image are:")
misalignment_angle.drawMatches(img5, kp5, img6, kp6, matches3)
print ("The misalignment of left image are:")
misalignment_angle.drawMatches(img7, kp7, img8, kp8, matches4)






