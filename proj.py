import cv2
import numpy as np
import math
import NM as NN
import time
import os
import copy

import sys

class Brojevi:
    def __init__(self,pozicija,number):
        self.pozicija = pozicija
        self.number = number

def provera_prelaza(br_linije,suma,x,y,w,h,x1,y1,x2,y2,idx_broja,broj):
    # print(x,y,w,h,'tacke')
    # print(x1,y1,x2,y2, 'tacke2')
    if x + w >= x1 and x <= x2 and y + h >= y2 and y <= y1:  # zivimo u nadi da nema su sve linije / ovako a da nema \ ovakvih
        # sad gledamo samo brojeve koji su u kvadratu oko linije
        #print('u kvadratu je i visina mu je:',h)
        X = x2 - x1
        Y = y1 - y2
        m = x - x1
        #n = y - y2
        xc = x
        yc = y1-((Y * (xc - x1)) / X)
        xz = xc + w
        yz = y1-((Y * (xz - x1)) / X)
       # print('y= ',y,' h=',h, 'yc= ',yc, ' yz= ',yz)
        if y <= yc and  y + h >= yz:
            #prelazi liniju
            if br_linije == 0:
                pomm = 0
                for pp in trenutno_na_linijim:
                    if pp == idx_broja:
                        pomm = 1
                        #print('fail -')
                if (pomm == 0):
                    suma -= broj
                    #print('Suma je umanjena za ',rez,' i sada iznosi: ',suma)
                   # print('-',broj,'     ',suma)
                    trenutno_na_linijim.append(idx_broja)
            else:
                pomm = 0
                for pp in trenutno_na_linijip:
                    if pp == idx_broja:
                        pomm = 1
                        #print('fail +')
                if (pomm == 0):
                    suma += broj
                    #print('Suma je uvecana za ', rez, ' i sada iznosi: ', suma)
                    #print('+', broj,'     ',suma)
                    trenutno_na_linijip.append(idx_broja)
    br_linije += 1
    return br_linije,suma


ann=NN.create_ann()
ann = NN.load_object('trainedb.god')
directory = 'videos'
vid = ['a.jpg','b.jpg','c.jpg','d.jpg','e.jpg','f.jpg','g.jpg','h.jpg','i.jpg','j.jpg']
w=-1
glob_tren= {}
br_videa=0
for imevidea in os.listdir(directory):
    w+=1
    if imevidea.endswith('.avi'):

        cap = cv2.VideoCapture(os.path.join(directory, imevidea))
        kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (5,5))
        #kernel = np.rot90(kernel)
        #HOUGH transform
        ret, imgorg = cap.read()
        visina, sirina,asaas = np.shape(imgorg)
        #ZA ZELENU LINIJU
        img = imgorg.copy()
        img[:,:,2] = 0
        img[:, :, 0] = 0
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150,apertureSize = 3)
        fr_d = cv2.dilate(edges, kernel, iterations=4)
        fr_e = cv2.erode(fr_d,kernel, iterations=3)
        fr_d = cv2.dilate(fr_e, kernel, iterations=1)
        fr_e = cv2.erode(fr_d,kernel, iterations=1)
        minLineLength = 250
        maxLineGap = 50
        lines = cv2.HoughLinesP(fr_e,1,np.pi/180,200,minLineLength,maxLineGap)
        pom_lines = []
        x1m,y1m,x2m,y2m = lines[0][0]
        for x1, y1, x2, y2 in lines[0]:
            if x1<x1m:
                x1m = x1
            if x2>x2m:
                x2m = x2
            if y1> y1m:
                y1m = y1
            if y2<y2m:
                y2m = y2
        xo = (x2m-x1m)
        yo= y1m-y2m
        xi=6
        yi=yo*xi/xo
        pom_lines.append([x1m+xi,y1m-yi,x2m-xi,y2m+yi])

        #ZA PLAVU LINIJU
        img = imgorg.copy()
        img[:, :, 1] = 0
        img[:, :, 2] = 0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 100, apertureSize=3)
        fr_d = cv2.dilate(edges, kernel, iterations=4)
        fr_e = cv2.erode(fr_d, kernel, iterations=4)
        fr_d = cv2.dilate(fr_e, kernel, iterations=1)
        fr_e = cv2.erode(fr_d, kernel, iterations=1)
        minLineLength = 250
        maxLineGap = 50
        lines = cv2.HoughLinesP(fr_e, 1, np.pi / 180, 200, minLineLength, maxLineGap)
        x1m, y1m, x2m, y2m = lines[0][0]
        for x1, y1, x2, y2 in lines[0]:
            if x1 < x1m:
                x1m = x1
            if x2 > x2m:
                x2m = x2
            if y1 > y1m:
                y1m = y1
            if y2 < y2m:
                y2m = y2
        xo = (x2m - x1m)
        yo = y1m - y2m
        xi = 4
        yi = yo * xi / xo
        pom_lines.append([x1m + xi, y1m - yi, x2m - xi, y2m + yi])

        #CRTANJE LINIJA NA SLIKU ZA HOUGH
        # sadada=-1
        # for x1,y1,x2,y2 in pom_lines:
        #     # print (x1,y1,x2,y2)
        #     sadada+=1
        #     if sadada ==0:
        #         cv2.line(imgorg,(x1,y1),(x2,y2),(0,0,128),2)
        #     else:
        #         cv2.line(imgorg, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv2.imwrite(vid[w],imgorg)
        #cv2.imshow('frame',img)
        #time.sleep(0.1)
        #fgbg = cv2.BackgroundSubtractorMOG()
        # UCITAVANJE ISTRENIRANE NM

        pom_frame = 0
        suma = 0
        neobradjeni = []
        viskovi = []
        obradjeni_sad = []
        trenutni = {}  # dictionary kljuc je index, vrednos je objekat Broj
        trenutni_pom = {}
        trenutno_na_linijim = []  # bice index odgovarajuceg iz trenutni za minus
        trenutno_na_linijip = []  # bice index odgovarajuceg iz trenutni za plus
        index = 0  # uvecavace se za svaki novi dodati
        alphabet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        sss=0
        while(cap.isOpened()):
            try:
                sss+=1
                viskovi = []
                neobradjeni = []
                obradjeni_sad = []
                trenutni_pom = {}
                ret, frame = cap.read()
                asa = cv2.cvtColor(frame, cv2.COLOR_BGR2XYZ)
                gray = cv2.cvtColor(asa, cv2.COLOR_BGR2GRAY)
                # za binarni threshold proeko 200 samo brojevi ostanu
                r, frame_bin =  cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                asas = NN.invert((NN.dilate(frame_bin)))
                selected_regions, numbers, pozicije = NN.select_roiV(frame.copy(),asas)
                #cv2.imshow('frame', selected_regions)
                #time.sleep(0.1)
                # if sss==55:
                #     print('da')
                #print(pozicije[0])
                if len(trenutni) == 0:  # prvi frejm
                    if br_videa==0:
                        for i in range(0, len(numbers)):
                            test_inputs = []
                            test_inputs.append(NN.matrix_to_vector(NN.scale_to_range(numbers[i])))
                            result = ann.predict(np.array(test_inputs, np.float32))
                            broj = NN.display_result(result, alphabet)
                            br = Brojevi(pozicije[i], broj[0])
                            #print('Datadi u prvom frejmu: ',broj[0])
                            trenutni[index] = br
                            index += 1
                    else:
                        trenutni = copy.deepcopy(glob_tren)
                        #print('odradjeno kopiranje')
                else:  # ostali frejmovi
                    trenutni_pom = copy.deepcopy(trenutni)
                    # if sss == 885:
                    #      print('a')
                    # print(len(trenutni), len(trenutni_pom))
                    for i in range(0, len(numbers)):
                        obradjen = 0
                        for idx_broja, broj in trenutni.items():
                            pom = 0
                            for sas in obradjeni_sad:  # za slucaj da su neka dva blizu
                                if sas == idx_broja:
                                    poma = 1
                            # da li je to taj iz prethodne iteracije
                            if pom == 0 and pozicije[i][0] >= trenutni[idx_broja].pozicija[0] - 2 and pozicije[i][1] >= \
                                            trenutni[idx_broja].pozicija[1] - 2 and pozicije[i][0] <= \
                                            trenutni[idx_broja].pozicija[0] + 5 and pozicije[i][1] <= \
                                            trenutni[idx_broja].pozicija[1] + 5:
                                if pozicije[i][0] != trenutni[idx_broja].pozicija[0] or pozicije[i][1] != \
                                        trenutni[idx_broja].pozicija[1]:  # updejt samo ako se nesto promenilo
                                    vek = [pozicije[i][0], pozicije[i][1], trenutni[idx_broja].pozicija[2],trenutni[idx_broja].pozicija[3]]
                                    # print('bio je trenutni:', trenutni[idx_broja].pozicija)
                                    # print('bio je trenutni pomocni:', trenutni_pom[idx_broja].pozicija)
                                    trenutni_pom[idx_broja].pozicija = vek
                                    # print('sad je trenutni:', trenutni[idx_broja].pozicija)
                                    # print('bio je pomocni:', trenutni_pom[idx_broja].pozicija)
                                obradjeni_sad.append(idx_broja)  # za kasniju proveru (mozda imaju neki koji su ispali iz kadra ili su sakriveni iza)
                                obradjen = 1
                                break
                        # Apdejtovani su svi sto su su bili u proslom koraku
                        # Sad se dodaju novi i proveravaju oni koji su nestali

                        if obradjen == 0:
                            # nije postojao do sada
                            if pozicije[i][0] <= 15 or pozicije[i][1] <= 15:
                                flag = 0
                                #sprecavam interpretaciju jednog broja na vise nacina
                                for trind, trbr in trenutni.items():
                                    if pozicije[i][0] >= trbr.pozicija[0] - trbr.pozicija[2]/2 and pozicije[i][0] + \
                                            pozicije[i][2] <= trbr.pozicija[0] + trbr.pozicija[2] + trbr.pozicija[2]/2 and \
                                            pozicije[i][1] >= trbr.pozicija[1] - trbr.pozicija[3]/2 and pozicije[i][1] + \
                                            pozicije[i][3] <= trbr.pozicija[1] + trbr.pozicija[3] + trbr.pozicija[3]/2:
                                        flag = 1
                                        break
                                # print('dodaje ga kao novi')
                                # sigurno je nov
                                if flag==0:
                                    test_inputs = []
                                    test_inputs.append(NN.matrix_to_vector(NN.scale_to_range(numbers[i])))
                                    result = ann.predict(np.array(test_inputs, np.float32))
                                    broj = NN.display_result(result, alphabet)
                                    br = Brojevi(pozicije[i], broj[0])
                                    #print('Dodat kao nov: ', broj[0])
                                    trenutni_pom[index] = br
                                    obradjeni_sad.append(index)
                                    obradjen = 1
                                    #cv2.waitKey(0)
                                    index += 1
                            # Brisanje onih sto izlaze iz ekrana

                        # sad su ostali oni sto su iza crte ili iza nekog drugog broja
                        if obradjen == 0:
                            viskovi.append(i)  # indeks u numbers i pozicije
                    neobradjeni = []
                    #print('viskovi',len(viskovi))
                    for ind, brr in trenutni.items():
                        pomm = 0
                        for inds in obradjeni_sad:
                            if (ind == inds):
                                pomm = 1
                                break
                        if pomm == 0:
                            neobradjeni.append(ind)
                    # sad imamo
                    # neobradjeni -> oni iz trenutnih koji nisu obradjeni, znaci da su skriveni iza ili prepolovljeni linijom
                    # viskovi -> dobijen region koji nigde ne pripada, nije nov nego ne deo nekog broja
                    obradjeni_viskovi = []
                    izbaceni = []
                    for indn in neobradjeni:
                        if trenutni[indn].pozicija[0] >= sirina - 35 or trenutni[indn].pozicija[1] >= visina - 35:
                            del trenutni_pom[indn]
                            #print('izbrisan')
                            obradjeni_sad.append(indn)
                            izbaceni.append(indn)
                            kl=-1
                            for visak in viskovi:
                                if pozicije[visak][0]>= trenutni[indn].pozicija[0]-trenutni[indn].pozicija[2] and pozicije[visak][1]>= trenutni[indn].pozicija[1]-trenutni[indn].pozicija[3]:
                                    obradjeni_viskovi.append(visak)
                                    kl=visak
                                    break
                            if kl!=-1:
                                viskovi.remove(kl)

                        else:
                            kl = -1
                            for indv in viskovi:
                                # xs = trenutni[indn].pozicija[2]/2
                                # ys = trenutni[indn].pozicija[3]/2
                                if trenutni[indn].pozicija[0]  <= pozicije[indv][0] + \
                                        pozicije[indv][2] and trenutni[indn].pozicija[1] <= \
                                                pozicije[indv][1] + pozicije[indv][3] and trenutni[indn].pozicija[0] + \
                                        2*trenutni[indn].pozicija[2] >= pozicije[indv][0] + pozicije[indv][2] and \
                                                                trenutni[indn].pozicija[1] + trenutni[indn].pozicija[3]*2 >= \
                                                        pozicije[indv][1] + pozicije[indv][3]:
                                    # posmatramo donju desnu tacku, mozda se pojavio ispod linije
                                    #print("MOZDA SE I OVO DESI NEKAD")
                                    vek = [pozicije[indv][0] + pozicije[indv][2] - trenutni[indn].pozicija[2],
                                           pozicije[indv][1] + pozicije[indv][3] - trenutni[indn].pozicija[3],
                                           trenutni[indn].pozicija[2], trenutni[indn].pozicija[3]]
                                    trenutni_pom[indn].pozicija = vek
                                    obradjeni_viskovi.append(indv)
                                    kl = indv
                                    obradjeni_sad.append(indn)
                                    break
                            if kl!= -1:
                                viskovi.remove(kl)
                    #print('obradjeni viskovi', len(obradjeni_viskovi))
                    #print('viskovi', len(viskovi))
                    neobradjeni = []
                    for ind, brr in trenutni.items():
                        pomm = 0
                        for inds in obradjeni_sad:
                            if (ind == inds):
                                pomm = 1
                                break
                        if pomm == 0:
                            neobradjeni.append(ind)
                    obradjeni_viskovi = []
                    #gledamo veliku okolinu broja iz viska tako da se poklapa sa nekim ko nije obradjen al poredimo i numbers
                    for visak in viskovi:
                        if (pozicije[visak][3] > 21 or pozicije[visak][2] > 17) and pozicije[visak][0]+pozicije[visak][2]< sirina-35 and pozicije[visak][1]+pozicije[visak][3]< visina-35:
                            for neob in neobradjeni:
                                if trenutni[neob].pozicija[0]<= pozicije[visak][0]+2.5*pozicije[visak][2] and \
                                    trenutni[neob].pozicija[0]+trenutni[neob].pozicija[2] >= pozicije[visak][0]-1.5*pozicije[visak][2] and \
                                        trenutni[neob].pozicija[1]<=pozicije[visak][1]+2.5*pozicije[visak][3] and \
                                        trenutni[neob].pozicija[1] + trenutni[neob].pozicija[3] >= pozicije[visak][1] - \
                                        1.5*pozicije[visak][3]:
                                    test_inputs = []
                                    test_inputs.append(NN.matrix_to_vector(NN.scale_to_range(numbers[visak])))
                                    result = ann.predict(np.array(test_inputs, np.float32))
                                    broj = NN.display_result(result, alphabet)
                                    if broj[0]==trenutni[neob].number:
                                        #pretpostavimo da je to taj
                                        vek = pozicije[visak]
                                        trenutni_pom[neob].pozicija = vek
                                        obradjeni_sad.append(neob)
                                        obradjeni_viskovi.append(visak)

                    neobradjeni = []
                    for ind, brr in trenutni.items():
                        pomm = 0
                        for inds in obradjeni_sad:
                            if (ind == inds):
                                pomm = 1
                                break
                        if pomm == 0:
                            neobradjeni.append(ind)


                    for neo in neobradjeni:
                        tren_najbolji = -1
                        rez_naj = 1000000
                        for indx, brt in trenutni.items():
                            plm=0
                            for iz in izbaceni:
                                if(indx == iz):
                                    plm=1
                            for iz in neobradjeni:
                                if(indx == iz):
                                    plm=1
                            if plm==0 and neo != indx:
                                # trazimo mu najblizi drugi broj da ga pomerimo koliko i njega, mozda ga preklapa
                                if ((trenutni[neo].pozicija[0] - trenutni[indx].pozicija[0]) ** 2 + (
                                    trenutni[neo].pozicija[1] - trenutni[indx].pozicija[1]) ** 2) < rez_naj:
                                    rez_naj = ((trenutni[neo].pozicija[0] - trenutni[indx].pozicija[0]) ** 2 + (
                                    trenutni[neo].pozicija[1] - trenutni[indx].pozicija[1]) ** 2)
                                    tren_najbolji = indx

                        if tren_najbolji != -1:
                            xp = trenutni_pom[tren_najbolji].pozicija[0] - trenutni[tren_najbolji].pozicija[0]  # razlika koliko se pomerio
                            yp = trenutni_pom[tren_najbolji].pozicija[1] - trenutni[tren_najbolji].pozicija[1]
                            #print('Pomeram ga za ',xp,yp)
                            vek = [trenutni[neo].pozicija[0] + xp, trenutni[neo].pozicija[1] + yp,
                                   trenutni[neo].pozicija[2], trenutni[neo].pozicija[3]]
                            trenutni_pom[neo].pozicija = vek

                    preostali_viskovi = []
                    for vis in viskovi:
                        p = 0
                        for ov in obradjeni_viskovi:
                            if ov == vis:
                                p = 1
                                break
                        if p == 0:
                            preostali_viskovi.append(vis)
                        #print('preostali viskovi', len(preostali_viskovi))
                    #sad cemo da vidimo da li neki od ovih mozda nije do sada sve vreme bio skriven pa se zato tek sad pojavio
                    #if len(trenutni_pom)>=len(pozicije):
                    for visak in preostali_viskovi:
                        #print('pore', pozicije[visak])
                        if (pozicije[visak][3] > 21 or pozicije[visak][2] > 17) and pozicije[visak][0]+pozicije[visak][2] < sirina-35 and pozicije[visak][1]+pozicije[visak][3] < visina -35:# to su mere u kojima ga moze prepoznati kao redovan broj, otp\
                            #i da se ne moze dodati u granici izbacivanja
                            #sledi provera da li je na pozicijama nekog drugog broja, ako je do sada bio iza, ne moze se naci ispred nekog nego je uvek iza
                            #sprecavamo da se jedan broj interpretira na dva nacina
                            #print('pozicija: ' ,pozicije[visak],sirina,visina)
                            flag=0
                            for trind, trbr in trenutni.items():
                                if pozicije[visak][0] >= trbr.pozicija[0]-3 and pozicije[visak][0]+pozicije[visak][2] <= trbr.pozicija[0]+trbr.pozicija[2]+3 and \
                                pozicije[visak][1] >= trbr.pozicija[1] - 3 and pozicije[visak][1] + pozicije[visak][3] <= trbr.pozicija[1] + trbr.pozicija[3] + 3:
                                    flag=1
                                    break
                            if flag==0:
                                test_inputs = []
                                test_inputs.append(NN.matrix_to_vector(NN.scale_to_range(numbers[visak])))
                                result = ann.predict(np.array(test_inputs, np.float32))
                                broj = NN.display_result(result, alphabet)
                                br = Brojevi(pozicije[visak], broj[0])
                                #print('dodao kao odkriveni',broj[0],pozicije[visak])
                                #cv2.waitKey(0)
                                trenutni_pom[index] = br
                                index += 1
                                #provera da li je mozda presao preko neke linije dok je bio sakriven

                                sadidx = index-1
                                tren_najbolji = -1
                                rez_naj = 1000000
                                for indx, brt in trenutni.items():
                                    plm = 0
                                    for iz in izbaceni:
                                        if (indx == iz):
                                            plm = 1

                                    if plm == 0 and sadidx != indx:
                                        # trazimo mu najblizi drugi broj da ga pomerimo koliko i njega, mozda ga preklapa
                                        if ((pozicije[visak][0] - trenutni[indx].pozicija[0]) ** 2 + (
                                                pozicije[visak][1] - trenutni[indx].pozicija[1]) ** 2) < rez_naj:
                                            rez_naj = ((pozicije[visak][0] - trenutni[indx].pozicija[0]) ** 2 + (
                                                    pozicije[visak][1] - trenutni[indx].pozicija[1]) ** 2)
                                            tren_najbolji = indx
                                if tren_najbolji != -1:
                                    minus = 0
                                    plus =0
                                    for idx in trenutno_na_linijim:
                                        if idx == tren_najbolji:
                                            minus=1
                                            break
                                    for idx in trenutno_na_linijip:
                                        if idx == tren_najbolji:
                                            plus = 1
                                            break
                                    # moram ih rucno sabrati jer su mozda vec presli liniju
                                    if minus !=0:
                                        suma-=broj[0]
                                        #print('-', broj[0],'  naknadno',suma)
                                        trenutno_na_linijim.append(sadidx)
                                    if plus !=0:
                                        suma+=broj[0]
                                        #print('+', broj[0],'  naknadno',suma)
                                        trenutno_na_linijip.append(sadidx)


                    trenutni = {}
                    trenutni = copy.deepcopy(trenutni_pom)
                    glob_tren={}
                    glob_tren = copy.deepcopy(trenutni_pom)
                        # print(len(numbers), len(trenutni))
                #trenutni = {}

                #print('trenutni na kraju, ', len(trenutni), 'pom', len(trenutni_pom))
                trenutni_pom = {}

                for inb, br in trenutni.items():
                    br_linije = 0
                    for x1, y1, x2, y2 in pom_lines:
                        x, y, w, h = trenutni[inb].pozicija
                        br_linije, suma = provera_prelaza(br_linije, suma, x, y, w, h, x1, y1, x2, y2, inb, trenutni[inb].number)
                #pom_frame += 1
                # if len(trenutni)!=len(numbers):
                #    print('razliciti su ',len(trenutni),len(numbers))
                #    for i,x in trenutni.items():
                #        print(x.pozicija)
                #    print('numb')
                #    for c in pozicije:
                #       print(c)
                #   time.sleep(500)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                print "Unexpected error:", sys.exc_info()[0]
                print('kraj videa')
                #br_videa+=1

                break
        print('KONACNA SUMA ',imevidea,' JE ', suma)

cap.release()
cv2.destroyAllWindows()


# hough transform za one dve linije
# img = cv2.imread('pom.png')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray,50,150,apertureSize = 3)
# minLineLength = 300
# maxLineGap = 100
# lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
# print(lines)
# for x1,y1,x2,y2 in lines[0]:
#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
#
# cv2.imwrite('houghlines5.jpg',img)

