from tkinter import *   
from time import time
import math
  
LaFenetre = Tk()  
LaFenetre.title("Goban")

LargeurFenetre=710
HauteurFenetre=710
LaFenetre.geometry(str(LargeurFenetre)+"x"+str(HauteurFenetre))  


tour="Noir"
debutX=38
debutY=39
espacementCase=34.5
Cx=debutX
Cy=debutY

pierresNoires=[]
positionsPierresNoires=[]
groupesNoirs=[]
pierresBlanches=[]
positionsPierresBlanches=[]
groupesBlancs=[]


def miseAJourGroupesNoirs() :
    global positionsPierresNoires, groupesNoirs
    positionDernierePierre=positionsPierresNoires[-1]
    haut=[positionDernierePierre[0]-1,positionDernierePierre[1]]
    bas=[positionDernierePierre[0]+1,positionDernierePierre[1]]
    gauche=[positionDernierePierre[0],positionDernierePierre[1]-1]
    droite=[positionDernierePierre[0],positionDernierePierre[1]+1]
    groupesConnectes=[]
    for groupe in groupesNoirs :
        if (haut in groupe) or (bas in groupe) or (gauche in groupe) or (droite in groupe) :
            groupesConnectes.append(groupe)
    groupeFusion=[positionDernierePierre]
    for groupe in groupesConnectes :
        groupeFusion=groupeFusion+groupe
        groupesNoirs.remove(groupe)
    groupesNoirs.append(groupeFusion)

def libertésPotentiellesGroupe(groupe) :
    libertes=[]
    for position in groupe :
        haut=[position[0]-1,position[1]]
        bas=[position[0]+1,position[1]]
        gauche=[position[0],position[1]-1]
        droite=[position[0],position[1]+1]
        if not((haut in groupe) or (haut in libertes)) :
            libertes.append(haut)
        if not((bas in groupe) or (bas in libertes)) :
            libertes.append(bas)
        if not((gauche in groupe) or (gauche in libertes)) :
            libertes.append(gauche)
        if not((droite in groupe) or (droite in libertes)) :
            libertes.append(droite)
    return libertes



def colonne(x) :
    global debutX
    candidatColonne=math.floor((x-debutX)/espacementCase)
    if candidatColonne<0 :
        return 0
    if candidatColonne>18 :
        return 18
    else :
        return candidatColonne

def ligne(y) :
    global debutY
    candidatLigne=math.floor((y-debutY)/espacementCase)
    if candidatLigne<0 :
        return 0
    if candidatLigne>18 :
        return 18
    else :
        return candidatLigne

def positionCaseX(x) :
    global debutX
    return ligne(x)*espacementCase+debutX

def positionCaseY(y) :
    global debutY
    return colonne(y)*espacementCase+debutY




def motion(event):
    global Cx, Cy
    if not(event.y<35 and event.x<35) : #Bug étrange de souris
        Cx=event.x+35
        Cy=event.y+35
        pierre.place(x=positionCaseX(Cx), y=positionCaseY(Cy))


def clique(event) :
    global tour
    if tour=="Noir" :
        pierresNoires.append(Label(LaFenetre, image=imageNoir, borderwidth=0))
        positionsPierresNoires.append([ligne(Cx),colonne(Cy)])
        pierresNoires[-1].place(x=positionCaseX(Cx), y=positionCaseY(Cy))
        miseAJourGroupesNoirs()

        pierre.configure(image=imageBlanc)
        tour="Blanc"
    else :
        pierresBlanches.append(Label(LaFenetre, image=imageBlanc, borderwidth=0))
        positionsPierresBlanches.append([ligne(Cx),colonne(Cy)])
        pierresBlanches[-1].place(x=positionCaseX(Cx), y=positionCaseY(Cy))
        pierre.configure(image=imageNoir)
        tour="Noir"
    
    print("Noir : " + str(positionsPierresNoires))
    print("Blanc : " + str(positionsPierresBlanches))
    print("Groupes noirs : " + str(len(groupesNoirs)))
    for groupe in groupesNoirs :
        print("lib pot : " + str(len(libertésPotentiellesGroupe(groupe))))
    



def retirerPierreNoire(ligne,colonne) :
    global pierresNoires, positionsPierresNoires
    indicePierre=positionsPierresNoires.index([ligne,colonne])
    pierresNoires[indicePierre].destroy()
    pierresNoires.remove(pierresNoires[indicePierre])
    positionsPierresNoires.remove([ligne,colonne])

def clique2(event) :
    global pierresNoires, positionsPierresNoires
    print(len(pierresNoires))
    retirerPierreNoire(3,3)
    print(len(pierresNoires))



imageFond=PhotoImage(file = r"goban.png")
imageBlanc=PhotoImage(file = r"blanc.png") 
imageNoir=PhotoImage(file = r"noir.png") 

fond=Label(LaFenetre, image=imageFond)
fond.place(x=30, y=30)

pierre=Label(LaFenetre, image=imageBlanc, borderwidth=0)
pierre.configure(image=imageNoir)



LaFenetre.bind("<Motion>",motion)
LaFenetre.bind("<Button-1>",clique)
LaFenetre.bind("<KeyPress-z>",clique2)


LaFenetre.mainloop()  



