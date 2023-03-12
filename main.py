import cv2
import requests
from datetime import datetime
import time
from function import *

#------------------------------ COSTANTI ------------------------------#

sFaceProtoPath = './opencv_face_detector.pbtxt'
sFaceModelPath = './opencv_face_detector_uint8.pb'
sGenderProtoPath = './gender_deploy.prototxt'
sGenderModelPath = './gender_net.caffemodel'
sAgeProtoPath = "./age_deploy.prototxt"
sAgeModelPath = "./age_net.caffemodel"
sURL = 'https://master.instanteat.it/it-it/admin/MPAiWeb/Action/TriggerEvent'
nId = 0
anModelMeanValues = (78.4263377603, 87.7689143744, 114.895847746)
asGenderList = ['maschio', 'femmina']
asAgeList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
             '(25-32)', '(38-43)', '(48-53)', '(60-100)']
csLocation = 'Ufficio'

# Load prediction model
oGenderNet = cv2.dnn.readNet(sGenderModelPath, sGenderProtoPath)
oFaceNet = cv2.dnn.readNet(sFaceModelPath, sFaceProtoPath)
oAgeNet = cv2.dnn.readNet(sAgeModelPath, sAgeProtoPath)

oWebcam = cv2.VideoCapture(0)
nPadding = 20

while cv2.waitKey(1) < 0:
    bStatus, oFrame = oWebcam.read()
    oSmallFrame = cv2.resize(oFrame, (0, 0), fx=0.5, fy=0.5)

    oFrameFace, aBboxes = get_face_box(oFaceNet, oSmallFrame)
    for aBox in aBboxes:
        oFace = oSmallFrame[max(0, aBox[1]-nPadding):min(aBox[3]+nPadding, oFrame.shape[0]-1),
                            max(0, aBox[0]-nPadding):min(aBox[2]+nPadding, oFrame.shape[1]-1)]
        oBlob = cv2.dnn.blobFromImage(
            oFace, 1.0, (227, 227), anModelMeanValues, swapRB=False)
        oGenderNet.setInput(oBlob)
        aGenderPreds = oGenderNet.forward()
        sGender = asGenderList[aGenderPreds[0].argmax()]
        oAgeNet.setInput(oBlob)
        aAgePreds = oAgeNet.forward()
        sAge = asAgeList[aAgePreds[0].argmax()]
        if(aGenderPreds[0].max() >= 0.95 and aAgePreds[0].max() >= 0.98):
            print("Genere: {}, confidenza = {:.3f}".format(
                sGender, aGenderPreds[0].max()))
            sLabel = "{}".format(sGender)
            cv2.putText(oFrameFace, sLabel, (aBox[0], aBox[1]-10),
                        cv2.FONT_ITALIC, 1.2, (0, 255, 255), 2)
            print("Età Output: {}".format(aAgePreds))
            print("Età: {}, confidenza = {:.3f}".format(
                sAge, aAgePreds[0].max()))
            sCurrentDate = datetime.now().strftime('%Y-%m-%d %H:%M')
            nId += 1
            aPayload = {'cWhen': sCurrentDate, 'nId': nId,
                        'csLocation': csLocation, 'csGender': sGender, 'csAgeInterval': sAge}
            sResponse = requests.post(sURL, data=aPayload)
            print(sResponse.text)
            time.sleep(4)
        cv2.imshow("Demo eta'-genere", oFrameFace)
oWebcam.release()
cv2.destroyAllWindows()
