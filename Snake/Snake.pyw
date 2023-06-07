from tkinter import *   
from time import time
  
LaFenetre = Tk()  
LaFenetre.title("Snake")

LargeurFenetre=1000
HauteurFenetre=900  
LaFenetre.geometry(str(LargeurFenetre)+"x"+str(HauteurFenetre))  

direction='d' #Aller à droite, par rapport aux touches ZQSD

imageCase=PhotoImage(file = r"C:\Users\Victor\Documents\Python\Snake\CaseNormale.png") 
imagePerdu=PhotoImage(file = r"C:\Users\Victor\Documents\Python\Snake\CasePerdu.png")

pixelsCase=25 #Taille en pixels des carrés du snake
nombreCases=10 #Taille du snake en nombre de carrés.
actuelPositionX=pixelsCase*(nombreCases-1)
actuelPositionY=0
positionsDesCases=[[pixelsCase*i,0] for i in range(0, nombreCases)] #liste des positions des cases
corpsDuSerpent=[] #déclaration du corps du serpent, au début composé d'aucune case
perdu=FALSE

for i in range(nombreCases):
    corpsDuSerpent.append(Label(LaFenetre, image=imageCase)) #déclaration des cases qui construisent le corps du serpent
    corpsDuSerpent[i].place(x=actuelPositionX, y=actuelPositionY) #premier placement des cases (redéfinies immédiatement par le premier décplacement)

#les fonctions à éxecuter quand une touche est pressée
def z_pressed(event):
    global direction
    direction='z'

def q_pressed(event):
    global direction
    direction='q'

def s_pressed(event):
    global direction
    direction='s'

def d_pressed(event):
    global direction
    direction='d'



#########################
def press(touche) :
    global actuelPositionY, actuelPositionX, pixelsCase, positionsDesCases
    if touche=='z' :
        actuelPositionY-=pixelsCase
    elif touche == 'q' :
        actuelPositionX-=pixelsCase
    elif touche == 's' :
        actuelPositionY+=pixelsCase
    elif touche =='d':
         actuelPositionX+=pixelsCase
    positionsDesCases.append( [actuelPositionX,actuelPositionY] )
    del positionsDesCases[0]
    deplacement(positionsDesCases)


def deplacement(liste) :
    for i in range(nombreCases):
        corpsDuSerpent[i].place(x=liste[i][0], y=liste[i][1]) #mise à jour du placement des cases après calcul des nouvelles positions


LaFenetre.bind("<KeyPress-z>",z_pressed)
LaFenetre.bind("<KeyPress-q>",q_pressed)
LaFenetre.bind("<KeyPress-s>",s_pressed)
LaFenetre.bind("<KeyPress-d>",d_pressed)

def checkPerdu() :
    global perdu, actuelPositionX, actuelPositionY
    for x in positionsDesCases[:-1] :
        if (x==positionsDesCases[-1]) :
            perdu=TRUE
    if (actuelPositionX<0 or actuelPositionX>LargeurFenetre or actuelPositionY<0 or actuelPositionY>HauteurFenetre) :
        perdu=TRUE

def passageDuTemps():
    global direction, positionsDesCases
    press(direction)
    
    checkPerdu()
  
    if perdu :
        nouvelleCasePerdu=Label(LaFenetre, image=imagePerdu)
        nouvelleCasePerdu.place(x=positionsDesCases[-1][0], y=positionsDesCases[-1][1])
    else :
        LaFenetre.after(100,passageDuTemps)




LaFenetre.after(100,passageDuTemps)
LaFenetre.mainloop()  



