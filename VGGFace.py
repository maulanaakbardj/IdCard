# ID Card Recognition
from matplotlib import pyplot as plt
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

faceDetector = MTCNN()

def extractFaces(imagePath,targetSize=(1280,1280)):
  img = plt.imread(imagePath)
  result = faceDetector.detect_faces(img)
  faces = []
  for i in range(len(result)):
    x1, y1, width, height = result[i]['box']
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    face = Image.fromarray(face)
    face = face.resize(targetSize)
    faceArray = asarray(face)
    faces.append(faceArray)
  return faces

def getFaceEmbedding(faces):
  faces = asarray(faces,"float32")
  preprocessFaces = preprocess_input(faces, version=2)
  model = VGGFace(model='resnet50', include_top=False, input_shape=(1280, 1280, 3), pooling='avg')
  faceEmbeddings = model.predict(preprocessFaces)
  return faceEmbeddings

Face_1 = extractFaces("Data/KTP1.JPG")
FaceEmbeddings_1 = getFaceEmbedding(Face_1)
Face_2 = extractFaces("Data/Selfie.JPG")
FaceEmbeddings_2 = getFaceEmbedding(Face_2)

def faceMatch(knownFace,knownFaceEmbeddings,testFace,testFaceEmbedding):
  score = cosine(knownFaceEmbeddings, testFaceEmbedding)
  if(score<=0.5):
    print("Face Match.. Score : "+str(score))
  else:
    print("Face Not Match.. Score : "+str(score))

faceMatch(Face_1,FaceEmbeddings_1,Face_2[0],FaceEmbeddings_2[0])
