from __future__ import print_function
#import potrebnih biblioteka

import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections
import pickle

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD

import matplotlib.pylab as pylab
# pylab.rcParams['figure.figsize'] = 16, 12 # za prikaz vecih slika i plotova, zakomentarisati ako nije potrebno
ann =[]
def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 25, 255, cv2.THRESH_BINARY)
    return image_bin
def invert(image):
    return 255-image
def display_image(image, color= False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
def dilate(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)
def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)

def select_roiV(image_orig, image_bin):
    '''Oznaciti regione od interesa na frejmu videa. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28.
        Za oznacavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obelezeni regioni
        i niz slika koje predstavljaju regione sortirane po rastucoj vrednosti x ose
    '''
    a, contours, hierarchy= cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    kot = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if h > 15 and w > 2 and h<27 and w <27 and area>60:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # oznaciti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
    regions_array = sorted(regions_array, key=lambda item: item[1][1])
    sorted_regions = [region[0] for region in regions_array]
    pozicije = [region[1] for region in regions_array]

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regions, pozicije

def select_roi(image_orig, image_bina):
    '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28.
        Za oznacavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obelezeni regioni
        i niz slika koje predstavljaju regione sortirane po rastucoj vrednosti x ose
    '''
    cv2.imwrite('PreFook.jpg', image_bina)
    a,contours, hierarchy= cv2.findContours(image_bina.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    #print('konture')
    #print('konture: ',len(contours))
    i =0
    cv2.imwrite('Fook.jpg', image_bina)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)

        if h>15 and w>2 and h< 40 and w<40 and area >135 and i<5000:#za obucavanje
        #if h>37: #za test
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # oznaciti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom

            if i >= 4500 and i<5000 and area > 200:
                #print('if',i)
                region = image_bina[y:y + h + 1, x:x + w + 1]
                regions_array.append([resize_region(region), (x, y, w, h)])
                cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 0), 2)
                i += 1
            elif i<4500:
               # print('else',i)
                region = image_bina[y:y + h + 1, x:x + w + 1]
                regions_array.append([resize_region(region), (x, y, w, h)])
                if i<500:
                    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    i+=1
                elif i<1000:
                    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    i += 1
                elif i <1500:
                    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    i += 1
                elif i < 2000:
                    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 255), 2)
                    i += 1
                elif i < 2500:
                    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    i += 1
                elif i < 3000:
                    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    i += 1
                elif i < 3500:
                    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255, 128, 255), 2)
                    i += 1
                elif i < 4000:
                    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (128, 0, 255), 2)
                    i += 1
                elif i < 4500:
                    cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 128, 255), 2)
                    i += 1
            #print('duzina regions_array: ',len(regions_array))


    regions_array = sorted(regions_array, key=lambda item: item[1][1]) #SORTIRAJ PO Y
    sorted_regions = [region[0] for region in regions_array]
    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regions

def select_roia(image_orig, image_bin):
    '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28.
        Za oznacavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obelezeni regioni
        i niz slika koje predstavljaju regione sortirane po rastucoj vrednosti x ose
    '''
    a,contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    #print('konture')
    #print(len(contours))
    i =0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)

        #if h>15 and w>2 and h< 40 and w<40 and area >135:#za obucavanje
        if h>37: #za test
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # oznaciti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            if i==0:
                cv2.rectangle(image_orig, (x, y), (x + w, y + h), (255, 0, 0), 2)
                i+=1
            elif i==1:
                cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
                i += 1
            elif i == 2:
                cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
                i =0


    regions_array = sorted(regions_array, key=lambda item: item[1][1]) #SORTIRAJ PO Y
    sorted_regions = sorted_regions = [region[0] for region in regions_array]
    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regions


def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255.
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    '''
    return image/255

def matrix_to_vector(image):
    '''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
    return image.flatten()


def prepare_for_ann(regions):
    '''Regioni su matrice dimenzija 28x28 ciji su elementi vrednosti 0 ili 255.
        Potrebno je skalirati elemente regiona na [0,1] i transformisati ga u vektor od 784 elementa '''
    ready_for_ann = []
    for region in regions:
        # skalirati elemente regiona
        # region sa skaliranim elementima pretvoriti u vektor
        # vektor dodati u listu spremnih regiona
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))

    return ready_for_ann

def convert_output(alphabet):
    '''Konvertovati alfabet u niz pogodan za obucavanje NM,
        odnosno niz ciji su svi elementi 0 osim elementa ciji je
        indeks jednak indeksu elementa iz alfabeta za koji formiramo niz.
        Primer prvi element iz alfabeta [1,0,0,0,0,0,0,0,0,0],
        za drugi [0,1,0,0,0,0,0,0,0,0] itd..
    '''
    nn_outputs = []
    # for index in range(len(alphabet)):
    #     output = np.zeros(len(alphabet))
    #     output[index] = 1
    #     nn_outputs.append(output)

    for index in range(0,10):
        for i in range(0,500):
            pom = np.zeros(10)
            pom[index]=1
            nn_outputs.append(pom)
   # print(nn_outputs)
    return np.array(nn_outputs)


def create_ann():
    '''Implementacija vestacke neuronske mreze sa 784 neurona na uloznom sloju,
        128 neurona u skrivenom sloju i 10 neurona na izlazu. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(10, activation='sigmoid'))
    return ann


def train_ann(ann, X_train, y_train):
    '''Obucavanje vestacke neuronske mreze'''
    X_train = np.array(X_train, np.float32)  # dati ulazi
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi za date ulaze
    print ('treniranje')
    print (len(X_train))
    print(len(y_train))
    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    # print(len(X_train))
    # print(len(y_train))
    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, epochs=2000, batch_size=1, verbose=0, shuffle=False)
    print("Zavrsio sa obucavanjem")
    save_object(ann,'trainedb.god')
    print("Sacuvao")

    return ann

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as input:
        #ann = create_ann()
        ann = pickle.load(input)
        return ann

def winner(output): # output je vektor sa izlaza neuronske mreze
    '''pronaci i vratiti indeks neurona koji je najvise pobudjen'''
    return max(enumerate(output), key=lambda x: x[1])[0]


def display_result(outputs, alphabet):
    '''za svaki rezultat pronaci indeks pobednickog
        regiona koji ujedno predstavlja i indeks u alfabetu.
        Dodati karakter iz alfabet u result'''
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result

def compare(filename,numbers):
    ann = load_object(filename)
    alphabet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    inputs = prepare_for_ann(numbers)
    result = ann.predict(np.array(inputs, np.float32))
    print(display_result(result, alphabet))
    return result
# TEST TEST TEST TEST TEST TEST TEST TEST
#
# with open('.theanorc', 'wb') as output:  # Overwrites any existing file.
#     pickle.dump('[global] floatX = float32 device = cuda0 [gpuarray] preallocate = 1', output, pickle.HIGHEST_PROTOCOL)

# image_color = load_image('digitss.png')
# img = invert(image_bin(image_gray(image_color)))
# cv2.imwrite('aaa.jpg',img)
# img_bin = erode(img)
# cv2.imwrite('aas.jpg',img_bin)
# selected_regions, numbers = select_roi(image_color, img)
# #display_image(selected_regions)
# cv2.imwrite('aa.jpg',numbers[4900])
# cv2.imwrite('aasa.jpg',selected_regions)
#
# #alphabet = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,6,6,6,6,6,7,7,7,7,7,8,8,8,8,8,9,9,9,9,9]*100
# # alphabet = [0]*500
# # alphabet.append([1]*500)
# # alphabet.append([2]*500)
# # alphabet.append([3]*500)
# # alphabet.append([4]*500)
# # alphabet.append([5]*500)
# # alphabet.append([6]*500)
# # alphabet.append([7]*500)
# # alphabet.append([8]*500)
# # alphabet.append([9]*500)
# alphabet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# #print(np.eye(10))
# print("duzina len")
# print(len(numbers))
# inputs = prepare_for_ann(numbers)
# outputs = convert_output(alphabet)
# ann = create_ann()
# ann = train_ann(ann, inputs, outputs)
#
# # result = ann.predict(np.array(inputs[2:4], np.float32))
# # print(result)
# # print(display_result(result, alphabet))
#
#
# #moj test
# # ann = create_ann()
# # ann = load_object('trainedb.god')
# test_color = load_image('test.jpg')
# test = invert(image_bin(image_gray(test_color)))
# test_bin = erode(dilate(test))
# selected_test, test_numbers = select_roia(test_color.copy(),test_bin)
# cv2.imwrite('as.jpg',test_bin)
# #print ((test_numbers))
# #display_image(selected_test)
# test_inputs = prepare_for_ann(test_numbers)
# result = ann.predict(np.array(test_inputs, np.float32))
# print(result)
# #alphabet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# print(display_result(result, alphabet))
